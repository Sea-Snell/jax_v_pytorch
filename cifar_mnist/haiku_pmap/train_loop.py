from copy import deepcopy
from functools import partial
from typing import Optional, Dict, Any
from micro_config import ConfigScript, ConfigScriptNoCache, MetaConfig
from dataclasses import dataclass, asdict
from collections import deque
import jax
import os
import pickle as pkl
import optax
import chex
from logs import label_logs, pool_logs, reduce_logs, log
from tqdm.auto import tqdm
import wandb
from haiku_configs import ConfigScriptModel, ConfigScriptOptim, ConfigScriptRNG
import json
from haiku_utils import batch_iterator, prefetch
from frozendict import frozendict

def unreplicate(tree):
  """
    Returns a single instance of a replicated array.
    source: https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#unreplicate
  """
  return jax.tree_map(lambda x: x[0], tree)

@dataclass
class StandardaEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    model: ConfigScriptModel
    rng: ConfigScriptRNG
    bsize: int
    prefetch_batches: Optional[int]
    eval_batches: Optional[int]
    loss_kwargs: Dict[str, Any]
    jit: bool
    sharded_model: bool

    def unroll(self, metaconfig: MetaConfig):
        devices = jax.local_devices()
        n_devices = len(devices)
        assert self.bsize % n_devices == 0, 'batch size must be divisible by number of devices'

        # conditionally block jit
        if not self.jit:
            fake_jit = chex.fake_jit()
            fake_jit.start()

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
        model, params, model_state = self.model.unroll(metaconfig)
        if not self.sharded_model:
            params = jax.device_put_replicated(params, devices)
            model_state = jax.device_put_replicated(model_state, devices)

        # define eval loss
        loss_kwargs = frozendict(self.loss_kwargs)
        @partial(jax.pmap, static_broadcasted_argnums=(0,5,), axis_name='data_local_device', devices=devices)
        def eval_loss(model, params, model_state, rng, loss_args, loss_kwargs):
            (_, logs,), _ = model.apply.loss(params, model_state, rng, *loss_args, train=False, **loss_kwargs)
            return logs
        
        # setup evaluator loop state
        eval_logs = []

        # eval on batches
        rng, new_rng = jax.random.split(rng)
        for i, items in enumerate(dataloader(new_rng)):
            
            # conditionally terminate early
            if self.eval_batches is not None and i >= self.eval_batches:
                break

            # shard data, get eval logs
            rng, *new_rng = jax.random.split(rng, n_devices+1)
            new_rng = jax.device_put_sharded(new_rng, devices)
            items = jax.tree_util.tree_map(lambda x: x.reshape(n_devices, x.shape[0] // n_devices, *x.shape[1:]), items)
            logs = eval_loss(model, params, model_state, new_rng, items, loss_kwargs)
            eval_logs.append(logs)
        
        # gather and postproc eval logs
        eval_logs = pool_logs(reduce_logs(eval_logs))

        # undo conditional jit block
        if not self.jit:
            fake_jit.stop()

        return eval_logs['loss'], eval_logs

@dataclass
class TrainLoop(ConfigScript):
    model: ConfigScriptModel
    train_data: ConfigScript
    optim: ConfigScriptOptim
    evaluator: StandardaEvaluator
    rng: ConfigScriptRNG
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    prefetch_batches: Optional[int]
    log_every: int
    eval_every: int
    save_every: Optional[int]
    jit: bool
    use_wandb: bool
    wandb_project: str
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):
        print('using config:', asdict(self))
        devices = jax.local_devices()
        n_devices = len(devices)
        assert self.bsize % n_devices == 0, 'batch size must be divisible by number of devices'
        print('using %d devices:' % (n_devices), devices)
        
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
        
        # conditionally block jit
        if not self.jit:
            fake_jit = chex.fake_jit()
            fake_jit.start()
        
        # get rng
        rng = self.rng.unroll(metaconfig)
        
        # setup dataset
        train_dataset = self.train_data.unroll(metaconfig)
        def dataloader(rng):
            iterator = batch_iterator(rng, train_dataset, self.bsize)
            if self.prefetch_batches is not None:
                iterator = prefetch(iterator, self.prefetch_batches)
            return iterator

        # setup training objects, replicate params
        model, params, model_state = self.model.unroll(metaconfig)
        optim, opt_state = self.optim.unroll(metaconfig)
        params, model_state = jax.device_put_replicated(params, devices), jax.device_put_replicated(model_state, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)

        # define training step
        loss_kwargs = frozendict(self.loss_kwargs)
        @partial(jax.pmap, static_broadcasted_argnums=(0,1,7,), axis_name='data_local_device', devices=devices)
        def step_fn(model, optim, params, model_state, opt_state, rng, loss_args, loss_kwargs):
            def grad_loss(params, model_state, rng, *loss_args, **loss_kwargs):
                (loss, logs), model_state = model.apply.loss(params, model_state, rng, *loss_args, **loss_kwargs)
                return loss, (logs, model_state,)
            (_, (logs, model_state,)), grads = jax.value_and_grad(grad_loss, has_aux=True)(params, model_state, rng, *loss_args, train=True, **loss_kwargs)
            grads = jax.lax.pmean(grads, axis_name='data_local_device')
            updates, opt_state = optim.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return logs, params, model_state, opt_state

        # initalize training loop state
        step = 0
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])
        evaluator = deepcopy(self.evaluator)
        assert evaluator.sharded_model, 'evaluator must allow sharded model'

        # train loop
        for epoch in tqdm(range(self.epochs)):
            rng, new_rng = jax.random.split(rng)
            for items in tqdm(dataloader(new_rng), total=(len(train_dataset) // self.bsize)):
                # step model, shard data, and get training logs
                rng, *new_rng = jax.random.split(rng, n_devices+1)
                new_rng = jax.device_put_sharded(new_rng, devices)
                items = jax.tree_util.tree_map(lambda x: x.reshape(n_devices, x.shape[0] // n_devices, *x.shape[1:]), items)
                logs, params, model_state, opt_state = step_fn(model, optim, params, model_state, opt_state, new_rng, items, loss_kwargs)
                train_logs.append(logs)
                
                # publish training logs
                if (step + 1) % self.log_every == 0:
                    logs = reduce_logs(train_logs)
                    logs = pool_logs(label_logs(logs, 'train', {'step': step, 'epoch': epoch}))
                    log(logs, self.use_wandb)
                
                # clear training logs
                if (step + 1) % self.optim.grad_accum_steps == 0:
                    train_logs = []
                
                # begin evaluation
                if (step + 1) % self.eval_every == 0:

                    # get eval logs
                    evaluator.model.params, evaluator.model.state = params, model_state
                    eval_perf, eval_logs = evaluator.unroll(metaconfig)

                    # publish eval logs
                    eval_logs = pool_logs(label_logs(eval_logs, 'eval', {'step': step, 'epoch': epoch}))
                    log(eval_logs, self.use_wandb)
                    
                    # conditionally save best model and optimizer state
                    if save_dir is not None and eval_perf < best_perf:
                        print('new best eval loss! Saving ...')
                        with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
                            pkl.dump(unreplicate((params, model_state,)), f)
                        with open(os.path.join(save_dir, 'optim.pkl'), 'wb') as f:
                            pkl.dump(unreplicate(opt_state), f)
                        print('saved.')
                        best_perf = eval_perf
                
                # periodically save checkpoint
                if save_dir is not None and self.save_every is not None and (step + 1) % self.save_every == 0:
                    print('saving checkpoint...')
                    
                    # conditionally delete old checkpoints
                    if (self.max_checkpoints is not None) and (len(saved_checkpoints) >= self.max_checkpoints):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    
                    # save
                    with open(os.path.join(save_dir, 'model_%d.pkl' % (step)), 'wb') as f:
                        pkl.dump(unreplicate((params, model_state,)), f)
                    saved_checkpoints.append(os.path.join(save_dir, 'model_%d.pkl' % (step)))
                    print('saved.')
                
                # increment step counter
                step += 1
                
                # conditionally terminate
                if self.max_steps is not None and step >= self.max_steps:
                    return
        
        # undo conditional jit block
        if not self.jit:
            fake_jit.stop()

