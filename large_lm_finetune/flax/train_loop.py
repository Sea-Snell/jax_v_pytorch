from copy import deepcopy
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
from base_configs import AdamWConfig, AdaFactorConfig
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

@dataclass
class StandardEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    model: PretrainedHFPjitModelConfig
    rng: ConfigScriptRNG
    bsize: int
    prefetch_batches: Optional[int]
    eval_batches: Optional[int]
    max_len: int
    pjit: bool
    loss_kwargs: Dict[str, Any]
    verbose: bool

    def unroll(self, metaconfig: MetaConfig):
        # get rng
        rng = self.rng.unroll(metaconfig)

        # setup dataset
        eval_dataset = self.eval_data.unroll(metaconfig)
        def dataloader(rng):
            iterator = batch_iterator(rng, eval_dataset, self.bsize)
            if self.prefetch_batches is not None:
                iterator = prefetch(iterator, self.prefetch_batches)
            return iterator

        # load model
        model, params, tokenizer, rules = self.model.unroll(metaconfig)
        if not model.config.is_encoder_decoder:
            position_ids = jnp.broadcast_to(jnp.arange(self.max_len)[None, :], (self.bsize, self.max_len))
        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # specifies how to split model parameters beteen devices
        param_spec = set_partitions(unfreeze(params), rules)

        # initialization function for splitting parameters to devices
        if self.pjit:
            p_get_initial_params = pjit(
                _id_fn, 
                in_axis_resources=(param_spec, None), 
                out_axis_resources=(param_spec, None), 
            )
        else:
           p_get_initial_params = _id_fn 

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        if self.verbose:
            print('using mesh shape:', mesh_devices.shape)
            print('full mesh:', mesh_devices)

        # split the params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            params, _ = p_get_initial_params(freeze(params), jnp.ones((), dtype=jnp.uint32))

        # define eval loss
        def t5_eval_loss(params, batch):
            batch = batch.unfreeze()
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            batch['decoder_attention_mask'] = decoder_attn_mask
            logits = model(**batch, params=params, train=False).logits
            _, logs = model_loss(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:], decoder_attn_mask[:, :-1])
            return logs
        
        def lm_eval_loss(params, batch):
            batch = batch.unfreeze()
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            logits = model(**batch, params=params, train=False).logits
            _, logs = model_loss(logits[:, :-1, :], batch['input_ids'][:, 1:], attn_mask[:, :-1])
            return logs
        
        if not model.config.is_encoder_decoder:
            eval_loss = lm_eval_loss
        else:
            eval_loss = t5_eval_loss
        
        if self.pjit:
            p_eval_loss = pjit(
                eval_loss, 
                in_axis_resources=(param_spec, None,), 
                out_axis_resources=(None), 
            )
        else:
            p_eval_loss = eval_loss
        
        # setup evaluator loop state
        eval_logs = []
        rng = self.rng.unroll(metaconfig)

        # eval on batches
        with Mesh(mesh_devices, ("dp", "mp")):
            rng, new_rng = jax.random.split(rng)
            for i, items in enumerate(dataloader(new_rng)):
                
                # conditionally terminate early
                if self.eval_batches is not None and i >= self.eval_batches:
                    break

                # get eval logs
                logs = p_eval_loss(params, items)
                eval_logs.append(logs)
        
        # gather and postproc eval logs
        eval_logs = pool_logs(reduce_logs(eval_logs))

        return eval_logs['loss'], eval_logs

@dataclass
class TrainLoop(ConfigScript):
    train_data: ConfigScript
    model: PretrainedHFPjitModelConfig
    optim: ConfigScript
    evaluator: StandardEvaluator
    rng: ConfigScriptRNG
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    max_len: int
    prefetch_batches: Optional[int]
    log_every: int
    eval_every: int
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
        if not model.config.is_encoder_decoder:
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
        if isinstance(self.optim, AdamWConfig):
            opt_state_spec, param_spec = jax.tree_map(
                get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
            )
        elif isinstance(self.optim, AdaFactorConfig):
            opt_state_spec, param_spec = jax.tree_map(
                get_opt_spec, state_shapes, is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,))
            )
            # opt_state_spec = opt_state_spec._replace(inner_opt_state=None)
            opt_state_spec = None
        else:
            raise NotImplementedError

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
        
        def get_params(rng):
            return model.init_weights(rng, (1, 1,))
        
        p_get_initial_state = pjit(
            get_params,
            in_axis_resources=(None,), 
            out_axis_resources=param_spec, 
        )

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)

        # split the opt_state and params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            # opt_state, params = p_get_initial_state(params)
            rng, new_rng = jax.random.split(rng)
            params = p_get_initial_state(new_rng)
        print(jax.tree_util.tree_map(lambda x: list(map(lambda y: y.shape, x.device_buffers)), params))
        print(jax.tree_util.tree_map(lambda x: x.shape, params))
        
        # # define lm training step
        # def lm_step_fn(params: PyTree, opt_state: PyTree, rng: jax.random.PRNGKey, batch: FrozenDict):
        #     batch = batch.unfreeze()
        #     attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
        #     batch['attention_mask'], batch['position_ids'] = attn_mask, position_ids
        #     def grad_loss(params: PyTree):
        #         logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
        #         loss, logs = model_loss(logits[:, :-1, :], batch['input_ids'][:, 1:], attn_mask[:, :-1])
        #         return loss, logs
        #     (_, logs), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        #     updates, opt_state = optim.update(grads, opt_state, params)
        #     params = optax.apply_updates(params, updates)
        #     return logs, params, opt_state
        
        # # define seq2seq training step
        # def t5_step_fn(params: PyTree, opt_state: PyTree, rng: jax.random.PRNGKey, batch: FrozenDict):
        #     batch = batch.unfreeze()
        #     attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
        #     batch['attention_mask'] = attn_mask
        #     decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
        #     batch['decoder_attention_mask'] = decoder_attn_mask
        #     def grad_loss(params: PyTree):
        #         logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
        #         loss, logs = model_loss(logits[:, :-1, :], batch['decoder_input_ids'][:, 1:], decoder_attn_mask[:, :-1])
        #         return loss, logs
        #     (_, logs), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        #     updates, opt_state = optim.update(grads, opt_state, params)
        #     params = optax.apply_updates(params, updates)
        #     return logs, params, opt_state
        
        # if not model.config.is_encoder_decoder:
        #     step_fn = lm_step_fn
        # else:
        #     step_fn = t5_step_fn

        # if self.pjit:
        #     p_step_fn = pjit(
        #         step_fn, 
        #         in_axis_resources=(param_spec, opt_state_spec, None, None), 
        #         out_axis_resources=(None, param_spec, opt_state_spec), 
        #         donate_argnums=(0, 1), 
        #     )
        # else:
        #     p_step_fn = step_fn

        # # initalize training loop state
        # train_logs = []
        # best_perf = float('inf')
        # saved_checkpoints = deque([])
        # rng = self.rng.unroll(metaconfig)
        # step = 0
        # evaluator = deepcopy(self.evaluator)

        # # train loop
        # with Mesh(mesh_devices, ("dp", "mp")):
        #     for epoch in tqdm(range(self.epochs)):
        #         rng, new_rng = jax.random.split(rng)
        #         for items in tqdm(dataloader(new_rng), total=(len(train_dataset) // self.bsize)):
                    
        #             # step model and get training logs
        #             rng, new_rng = jax.random.split(rng)
        #             logs, params, opt_state = p_step_fn(params, opt_state, new_rng, items)
        #             train_logs.append(logs)
                    
        #             # publish training logs
        #             if (step + 1) % self.log_every == 0:
        #                 logs = reduce_logs(train_logs)
        #                 logs = pool_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
        #                 if jax.process_index() == 0:
        #                     log(logs, self.use_wandb)
                    
        #             # clear training logs
        #             if (step + 1) % self.optim.grad_accum_steps == 0:
        #                 train_logs = []
                    
        #             # begin evaluation
        #             if (step + 1) % self.eval_every == 0:

        #                 # get eval logs
        #                 evaluator.model.params = params
        #                 eval_perf, eval_logs = evaluator.unroll(metaconfig)

        #                 # publish eval logs
        #                 eval_logs = pool_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch}))
        #                 if jax.process_index() == 0:
        #                     log(eval_logs, self.use_wandb)

        #                 # conditionally save best model and optimizer state
        #                 if save_dir is not None and eval_perf < best_perf:
        #                     if jax.process_index() == 0:
        #                         print('new best model! Saving ...')
        #                         model_dir = os.path.join(save_dir, 'model')
        #                         model.save_pretrained(
        #                             model_dir, 
        #                             params=params, 
        #                         )
        #                         print('saved.')
        #                         best_perf = eval_perf
                    
        #             # periodically save checkpoint
        #             if save_dir is not None and self.save_every is not None and (step + 1) % self.save_every == 0:
        #                 if jax.process_index() == 0:
        #                     print('saving checkpoint...')

        #                     # conditionally delete old checkpoints
        #                     if (self.max_steps is not None) and (len(saved_checkpoints) >= self.max_steps):
        #                         os.system('rm -rf %s' % (saved_checkpoints.popleft()))

        #                     model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
        #                     model.save_pretrained(
        #                         model_dir, 
        #                         params=params, 
        #                     )
        #                     saved_checkpoints.append(model_dir)
        #                 print('saved.')

        #             # conditionally terminate
        #             if self.max_steps is not None and (step + 1) >= self.max_steps:
        #                 return

        #             step += 1
