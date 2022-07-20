from functools import partial
from typing import Optional, Dict, Any
from micro_config import ConfigScript, MetaConfig, ConfigScriptNoCache
from dataclasses import dataclass, asdict
from flax_hf_configs import ConfigScriptRNG
from flax_utils import rngs_from_keys, batch_iterator, prefetch
from collections import deque, namedtuple
import jax
import os
import pickle as pkl
from model_configs.hf_model import PretrainedHFPjitModelConfig
from load_model_utils import set_partitions
from src import model_loss
from logs import reduce_logs, label_logs, pool_logs, log
from tqdm.auto import tqdm
import wandb
from flax.core.frozen_dict import freeze
from flax.serialization import to_bytes
import json
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
import optax
from jax.experimental.pjit import pjit
from load_model_utils import _id_fn
from jax.experimental.maps import Mesh
import numpy as np
from flax.training.train_state import TrainState
from jaxtyping import PyTree
from model_configs.hf_t5_remat import FlaxT5ForConditionalGeneration
from optax import MultiStepsState

@dataclass
class TrainLoop(ConfigScript):
    train_data: ConfigScript
    model: PretrainedHFPjitModelConfig
    optim: ConfigScript
    rng: ConfigScriptRNG
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    max_len: int
    prefetch_batches: Optional[int]
    log_every: int
    save_every: Optional[int]
    pjit: bool
    use_wandb: bool
    wandb_project: str
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):
        print('using config:', asdict(self))
        
        # save configs
        save_dir = metaconfig.convert_path(self.save_dir)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(asdict(self), f)
            with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
                pkl.dump(self, f)
        
        # initalize wandb
        if self.use_wandb:
            wandb.init(project=self.wandb_project, config=asdict(self))
        
        # get rng
        rng = self.rng.unroll(metaconfig)
        
        # setup dataset
        train_dataset = self.train_data.unroll(metaconfig)
        def dataloader(rng):
            iterator = batch_iterator(rng, train_dataset, self.bsize)
            if self.prefetch_batches is not None:
                iterator = prefetch(iterator, self.prefetch_batches)
            return iterator

        # setup training objects
        model, params, tokenizer, rules = self.model.unroll(metaconfig)
        optim = self.optim.unroll(metaconfig)
        if not isinstance(model, FlaxT5ForConditionalGeneration):
            position_ids = jnp.broadcast_to(jnp.arange(self.max_len)[None, :], (self.bsize, self.max_len))
        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # Shard params and optimizer state onto devices
        # Source: https://github.com/huggingface/transformers/blob/main/examples/research_projects/jax-projects/model_parallel/run_clm_mp.py
        def get_initial_state(params):
            opt_state = optim.init(params)
            return opt_state, params

        # specifies how to split model parameters beteen devices
        param_spec = set_partitions(unfreeze(params), rules)

        # Get the PyTree for opt_state, we don't actually initialize the opt_state yet.
        class ShapeDtype(object):
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
        params_shapes = jax.tree_map(lambda x: ShapeDtype(x.shape, x.dtype), params)
        state_shapes = jax.eval_shape(get_initial_state, params_shapes)

        # get PartitionSpec for opt_state, this is very specific to adamw
        # TODO: optax returns different state for different optimizers, how can we handle this generically ?
        # or maybe we don't since in our examples we just use adamw or adafactor
        def get_opt_spec(x):
            if isinstance(x, (dict, FrozenDict,)):
                return param_spec
            return None

        opt_state_spec, param_spec = jax.tree_map(
            get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
        )

        # pjit the get_initial_state function to shard params and init
        # optimizer state in sharded way
        if self.pjit:
            p_get_initial_state = pjit(
                get_initial_state,
                in_axis_resources=(param_spec,), 
                out_axis_resources=(opt_state_spec, param_spec),
            )
        else:
            p_get_initial_state = get_initial_state

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)

        # split the opt_state and params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            opt_state, params = p_get_initial_state(params)
        
        # define lm training step
        def lm_step_fn(params: PyTree, opt_state: PyTree, rng: jax.random.PRNGKey, batch: FrozenDict):
            batch = batch.unfreeze()
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'], batch['position_ids'] = attn_mask, position_ids
            def grad_loss(params: PyTree):
                logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
                loss, logs = model_loss(logits[:, :-1, :], batch['input_ids'][:, 1:], attn_mask[:, :-1])
                return loss, logs
            (_, logs), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return logs, params, opt_state
        
        # define seq2seq training step
        def t5_step_fn(params: PyTree, opt_state: PyTree, rng: jax.random.PRNGKey, batch: FrozenDict):
            batch = batch.unfreeze()
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            batch['decoder_attention_mask'] = decoder_attn_mask
            def grad_loss(params: PyTree):
                logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
                loss, logs = model_loss(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:], decoder_attn_mask[:, :-1])
                return loss, logs
            (_, logs), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return logs, params, opt_state
        
        if not isinstance(model, FlaxT5ForConditionalGeneration):
            step_fn = lm_step_fn
        else:
            step_fn = t5_step_fn

        if self.pjit:
            p_step_fn = pjit(
                step_fn, 
                in_axis_resources=(param_spec, opt_state_spec, None, None), 
                out_axis_resources=(None, param_spec, opt_state_spec), 
                donate_argnums=(0, 1), 
            )
        else:
            p_step_fn = step_fn

        # initalize training loop state
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])
        rng = self.rng.unroll(metaconfig)
        step = 0

        # train loop
        with Mesh(mesh_devices, ("dp", "mp")):
            for epoch in tqdm(range(self.epochs)):
                rng, new_rng = jax.random.split(rng)
                for items in tqdm(dataloader(new_rng), total=(len(train_dataset) // self.bsize)):
                    
                    # step model and get training logs
                    rng, new_rng = jax.random.split(rng)
                    logs, params, opt_state = p_step_fn(params, opt_state, new_rng, items)
                    train_logs.append(logs)
                    
                    # publish training logs
                    if (step + 1) % self.log_every == 0:
                        logs = reduce_logs(train_logs)
                        logs = pool_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                        log(logs, self.use_wandb)
                    
                    # clear training logs
                    if (step + 1) % self.optim.grad_accum_steps == 0:
                        train_logs = []
                    
                    # periodically save checkpoint
                    if save_dir is not None and self.save_every is not None and (step + 1) % self.save_every == 0:
                        if jax.process_index() == 0:
                            print('saving checkpoint...')

                            # conditionally delete old checkpoints
                            if (self.max_steps is not None) and (len(saved_checkpoints) >= self.max_steps):
                                os.system('rm -rf %s' % (saved_checkpoints.popleft()))

                            model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
                            model.save_pretrained(
                                model_dir, 
                                params=params, 
                            )
                            saved_checkpoints.append(model_dir)
                        print('saved.')

                    # conditionally terminate
                    if self.max_steps is not None and (step + 1) >= self.max_steps:
                        return

                    step += 1
