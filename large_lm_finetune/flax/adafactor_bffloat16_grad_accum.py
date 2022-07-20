# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformation wrappers."""

import functools
from typing import Any, Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src.wrappers import ShouldSkipUpdateFunction, MultiStepsState, _zeros_tree_like
Array = jnp.ndarray

class Bf16AdaMultiSteps:
  """An optimiser wrapper to spread gradient computation over multiple steps.

  This wrapper will allow multiple mini-steps to accumulate their gradients
  together before applying them. It wraps another optimiser, and makes sure that
  this optimiser updates its state only when enough mini-steps have been
  performed. At any other mini-step, the inner optimiser is not used and the
  updates returned by the wrapper are all 0.

  The number of mini-steps per gradient update is controlled by a function, and
  it can vary over training. This offers a mean of varying batch size over
  training.
  """

  def __init__(
      self,
      opt: base.GradientTransformation,
      every_k_schedule: Union[int, Callable[[Array], Array]],
      use_grad_mean: bool = True,
      should_skip_update_fn: Optional[ShouldSkipUpdateFunction] = None):
    """Initialiser.

    Args:
      opt: the wrapped optimiser.
      every_k_schedule: an int or f a function.
        * As a function, it returns how many mini-steps should be accumulated
          in a single gradient step. Its only argument is the current
          gradient step count. By varying the returned value, users can vary the
          overall training batch size.
        * If an `int`, this is the constant number of mini-steps per gradient
          update.
      use_grad_mean: if `True` (the default), gradients accumulated over
        multiple mini-steps are averaged. Otherwise, they are summed.
      should_skip_update_fn: if provided, this function is used to decide when
        to accept or reject the updates from a mini-step. When a mini-step is
        rejected, the inner state of `MultiSteps` is not updated. In other
        words, it is as if this mini-step never happened. For example:
        * to ignore updates containing inf or NaN, do
          `should_skip_update_fn=skip_not_finite`;
        * to ignore updates with a norm square larger then 42, do
          `should_skip_update_fn=functools.partial(skip_large_updates,
                                                   max_norm_sq=42.)`.
        Note that the optimiser's state `MultiStepsState` contains a field
        `skip_state` in which debugging and monitoring information returned
        by `should_skip_update_fn` is written.
    """
    self._opt = opt
    if isinstance(every_k_schedule, int):
      self._every_k_schedule = lambda step: every_k_schedule
    else:
      self._every_k_schedule = every_k_schedule
    self._use_grad_mean = use_grad_mean

    if self._use_grad_mean:
      # Use Welford algorithm for numerically stable aggregation of mean.
      self._acc_update = (
          lambda grad, acc, *, n_acc: acc + (grad - acc) / (n_acc + 1))
    else:
      self._acc_update = lambda grad, acc, *, n_acc: grad + acc

    if should_skip_update_fn is None:

      def should_skip_update_fn(*unused_args, **unused_kwargs):
        return jnp.array(False, dtype=jnp.bool_), ()

    self._should_skip_update_fn = should_skip_update_fn

  @property
  def inner_opt(self):
    return self._opt

  def init(self, params: Any) -> MultiStepsState:
    """Builds and returns initial `MultiStepsState`."""
    updates = _zeros_tree_like(params)
    gradient_step = jnp.zeros([], dtype=jnp.int32)
    _, skip_state = self._should_skip_update_fn(updates, gradient_step, params)
    init_state = MultiStepsState(
        mini_step=jnp.zeros([], dtype=jnp.int32),
        gradient_step=gradient_step,
        inner_opt_state=self._opt.init(params),
        acc_grads=updates,
        skip_state=skip_state)
    return init_state

  def update(self,
             updates: base.Updates,
             state: MultiStepsState,
             params: Optional[base.Params] = None
             ) -> Tuple[base.Updates, MultiStepsState]:
    """Accumulates gradients and proposes non-zero updates every `k_steps`."""
    k_steps = self._every_k_schedule(state.gradient_step)
    acc_grads = jax.tree_util.tree_map(
        functools.partial(self._acc_update, n_acc=state.mini_step),
        updates, state.acc_grads)

    should_skip_update, skip_state = self._should_skip_update_fn(
        updates, state.gradient_step, params)

    p_dtype = jax.tree_util.tree_leaves(params)[0].dtype

    def final_step(args):
      del args
      final_updates, new_inner_state = self._opt.update(
          acc_grads, state.inner_opt_state, params=params)
      final_updates = jax.tree_util.tree_map(lambda x: x.astype(p_dtype), final_updates)
      new_state = MultiStepsState(
          mini_step=jnp.zeros([], dtype=jnp.int32),
          gradient_step=numerics.safe_int32_increment(state.gradient_step),
          inner_opt_state=new_inner_state,
          acc_grads=_zeros_tree_like(acc_grads),
          skip_state=skip_state)
      return final_updates, new_state

    def mid_step(args):
      del args
      updates_shape_dtype, _ = jax.eval_shape(
          self._opt.update, acc_grads, state.inner_opt_state, params=params)
      # mid_updates = jax.tree_util.tree_map(
      #     lambda sd: jnp.zeros(sd.shape, sd.dtype), updates_shape_dtype)
      mid_updates = jax.tree_util.tree_map(
          lambda sd: jnp.zeros(sd.shape, p_dtype), updates_shape_dtype)
      new_state = MultiStepsState(
          mini_step=numerics.safe_int32_increment(state.mini_step),
          gradient_step=state.gradient_step,
          inner_opt_state=state.inner_opt_state,
          acc_grads=acc_grads,
          skip_state=skip_state)
      return mid_updates, new_state

    new_updates, new_state = jax.lax.cond(
        state.mini_step < k_steps - 1, (), mid_step, (), final_step)

    if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
      raise ValueError(
          'The `should_skip_update_fn` function should return a boolean scalar '
          f'array, but it returned an array of dtype {should_skip_update.dtype}'
          f' and shape {should_skip_update.shape}')

    multi_state_when_skip = MultiStepsState(
        mini_step=state.mini_step,
        gradient_step=state.gradient_step,
        inner_opt_state=state.inner_opt_state,
        acc_grads=state.acc_grads,
        skip_state=skip_state)
    zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
    new_updates, new_state = jax.lax.cond(
        should_skip_update,
        (), lambda args: (zero_updates, multi_state_when_skip),
        (), lambda args: (new_updates, new_state))

    return new_updates, new_state

  def has_updated(self, state: MultiStepsState) -> Array:
    return jnp.logical_and(state.mini_step == 0, state.gradient_step > 0)

  def gradient_transformation(self) -> base.GradientTransformation:
    return base.GradientTransformation(init=self.init, update=self.update)
