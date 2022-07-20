from typing import Union, Any, Dict, List, Tuple, Set, FrozenSet, Iterator, Optional, Callable
import itertools
import jax
import numpy as np
import collections
from jaxtyping import PyTree

def rngs_from_keys(rng: jax.random.KeyArray, keys: Union[List[str], Set[str], Tuple[str], FrozenSet[str]]) -> Dict[str, jax.random.KeyArray]:
    rngs = {}
    for k in keys:
        rng, new_rng = jax.random.split(rng)
        rngs[k] = new_rng
    return rngs

def split_rng_pytree(rng_pytree: PyTree[jax.random.KeyArray], splits: int=2) -> PyTree[jax.random.KeyArray]:
    if len(jax.tree_util.tree_leaves(rng_pytree)) == 0:
        return tuple([rng_pytree for _ in range(splits)])
    outer_tree_def = jax.tree_util.tree_structure(rng_pytree)
    split_rngs = jax.tree_util.tree_map(lambda x: tuple(jax.random.split(x, splits)), rng_pytree)
    return jax.tree_util.tree_transpose(outer_tree_def, jax.tree_util.tree_structure(tuple([0 for _ in range(splits)])), split_rngs)

def batch_idxs(rng: jax.random.KeyArray, data_size: int, bsize: int) -> np.ndarray:
    steps_per_epoch = data_size // bsize
    permutations = np.asarray(jax.random.permutation(rng, data_size))
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    return permutations

def batch_iterator(rng: jax.random.KeyArray, dataset: Any, bsize: int, postproc_f: Optional[Callable]=None) -> Iterator:
    if postproc_f is None:
        postproc_f = lambda x: x
    batches = batch_idxs(rng, len(dataset), bsize)
    for idxs in batches:
        yield postproc_f(dataset[idxs])

def prefetch(iterator: Iterator, queue_size: int = 2) -> Iterator:
    # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    # queue_size = 2 should be ok for one GPU.

    queue = collections.deque()

    def enqueue(n):
        for item in itertools.islice(iterator, n):
            queue.append(item)

    enqueue(queue_size)
    while queue:
        yield queue.popleft()
        enqueue(1)
