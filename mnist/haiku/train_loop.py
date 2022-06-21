from functools import partial
from typing import Optional, Dict, Any
from micro_config import ConfigScript, ConfigScriptNoCache, MetaConfig
from dataclasses import dataclass, asdict
from collections import deque
import jax
import numpy as np
import os
import pickle as pkl
import optax
import chex
from logs import label_logs, pool_logs, reduce_logs, log
from tqdm.auto import tqdm
import wandb
from haiku_configs import ConfigScriptModel, ConfigScriptOptim, ConfigScriptRNG
import json

@dataclass
class StandardaEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    model: ConfigScriptModel
    rng: ConfigScriptRNG
    bsize: int
    eval_batches: Optional[int]
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):

        # get rng
        rng = self.rng.unroll(metaconfig)

        # setup dataset
        eval_dataset = self.eval_data.unroll(metaconfig)
        steps_per_epoch = len(eval_dataset) // self.bsize
        
        # get batch indexes
        rng, new_rng = jax.random.split(rng)
        permutations = np.asarray(jax.random.permutation(new_rng, len(eval_dataset)))
        permutations = permutations[:steps_per_epoch * self.bsize]
        permutations = permutations.reshape(steps_per_epoch, self.bsize)

        # load model
        model, params, model_state = self.model.unroll(metaconfig)

        # define eval loss
        @partial(jax.jit, static_argnames=list(self.loss_kwargs.keys()))
        def eval_loss(params, model_state, rng, *args, **kwargs):
            (_, logs,), _ = model.apply.loss(params, model_state, rng, *args, train=False, **kwargs)
            return logs
        
        # setup evaluator loop state
        eval_logs = []

        # eval on batches
        for i, idxs in tqdm(enumerate(permutations)):
            items = eval_dataset[idxs]
            
            # conditionally terminate early
            if self.eval_batches is not None and i >= self.eval_batches:
                break

            # get eval logs
            rng, new_rng = jax.random.split(rng)
            logs = eval_loss(params, model_state, new_rng, *items, **self.loss_kwargs)
            eval_logs.append(logs)
        
        # gather and postproc eval logs
        eval_logs = pool_logs(reduce_logs(eval_logs))

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
    log_every: int
    eval_every: int
    save_every: Optional[int]
    jit: bool
    use_wandb: bool
    wandb_project: str
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):
        print('using config:', asdict(self))
        print('using device:', jax.devices()[0])
        
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
        steps_per_epoch = len(train_dataset) // self.bsize

        # setup training objects
        model, params, model_state = self.model.unroll(metaconfig)
        optim, opt_state = self.optim.unroll(metaconfig)

        # define training step
        @partial(jax.jit, static_argnums=(0,), static_argnames=list(self.loss_kwargs.keys()))
        def step_fn(model, params, model_state, opt_state, rng, *args, **kwargs):
            def grad_loss(params, model_state, rng, *args, **kwargs):
                (loss, logs), model_state = model.apply.loss(params, model_state, rng, *args, **kwargs)
                return loss, (logs, model_state,)
            (_, (logs, model_state,)), grads = jax.value_and_grad(grad_loss, has_aux=True)(params, model_state, rng, *args, train=True, **kwargs)
            updates, opt_state = optim.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return logs, params, model_state, opt_state

        # initalize training loop state
        step = 0
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])

        # train loop
        for epoch in tqdm(range(self.epochs)):

            # get batch indexes
            rng, new_rng = jax.random.split(rng)
            permutations = np.asarray(jax.random.permutation(new_rng, len(train_dataset)))
            permutations = permutations[:steps_per_epoch * self.bsize]
            permutations = permutations.reshape(steps_per_epoch, self.bsize)

            for idxs in tqdm(permutations):
                items = train_dataset[idxs]

                # step model and accumulate training logs
                rng, new_rng = jax.random.split(rng)
                logs, params, model_state, opt_state = step_fn(model, params, model_state, opt_state, new_rng, *items, **self.loss_kwargs)
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
                    self.evaluator.model.params, self.evaluator.model.state = params, model_state
                    eval_perf, eval_logs = self.evaluator.unroll(metaconfig)

                    # publish eval logs
                    eval_logs = pool_logs(label_logs(eval_logs, 'eval', {'step': step, 'epoch': epoch}))
                    log(eval_logs, self.use_wandb)
                    
                    # conditionally save best model and optimizer state
                    if save_dir is not None and eval_perf < best_perf:
                        print('new best eval loss! Saving ...')
                        with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
                            pkl.dump((params, model_state,), f)
                        with open(os.path.join(save_dir, 'optim.pkl'), 'wb') as f:
                            pkl.dump(opt_state, f)
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
                        pkl.dump((params, model_state,), f)
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

