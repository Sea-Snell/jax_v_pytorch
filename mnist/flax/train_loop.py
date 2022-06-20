from functools import partial
from typing import Optional, Dict, Any
from micro_config import ConfigScript, MetaConfig, ConfigScriptNoCache
from dataclasses import dataclass, asdict
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from flax_configs import TrainStateConfig, ConfigScriptModel, ConfigScriptRNG
from flax_utils import rngs_from_keys
from collections import deque
import jax
import os
import pickle as pkl
import chex
from logs import reduce_logs, label_logs, pool_logs, log
from tqdm.auto import tqdm
import wandb
from flax.core.frozen_dict import freeze
from flax.serialization import to_bytes
import json

@dataclass
class StandardEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    model: ConfigScriptModel
    rng: ConfigScriptRNG
    bsize: int
    eval_batches: Optional[int]
    dataloader_workers: int
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):

        # setup dataloader
        eval_dataset = self.eval_data.unroll(metaconfig)
        train_data_loader_kwargs = {'num_workers': self.dataloader_workers, 
                                    'batch_size': self.bsize, 
                                    'collate_fn': eval_dataset.collate}
        if not isinstance(eval_dataset, IterableDataset):
            train_data_loader_kwargs['shuffle'] = True
        eval_dataloader = DataLoader(eval_dataset, **train_data_loader_kwargs)

        # load model
        model, variables, rng_keys = self.model.unroll(metaconfig)

        # define eval loss
        @partial(jax.jit, static_argnames=list(self.loss_kwargs.keys()))
        def eval_loss(variables, rngs, *args, **kwargs):
            _, logs = model.apply(variables, *args, method=model.loss, rngs=rngs, 
                                  mutable=False, train=False, **kwargs)
            return logs
        
        # setup evaluator loop state
        eval_logs = []
        rng = self.rng.unroll(metaconfig)

        # eval on batches
        for i, items in tqdm(enumerate(eval_dataloader)):
            
            # conditionally terminate early
            if self.eval_batches is not None and i >= self.eval_batches:
                break

            # get eval logs
            rng, new_rng = jax.random.split(rng)
            rngs = rngs_from_keys(new_rng, rng_keys)
            logs = eval_loss(variables, rngs, *items, **self.loss_kwargs)
            eval_logs.append(logs)
        
        # gather and postproc eval logs
        eval_logs = pool_logs(reduce_logs(eval_logs))

        return eval_logs['loss'], eval_logs

@dataclass
class TrainLoop(ConfigScript):
    train_data: ConfigScript
    train_state: TrainStateConfig
    evaluator: StandardEvaluator
    rng: ConfigScriptRNG
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    log_every: int
    eval_every: int
    save_every: Optional[int]
    dataloader_workers: int
    jit: bool
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
        
        # conditionally block jit
        if not self.jit:
            fake_jit = chex.fake_jit()
            fake_jit.start()
        
        # setup dataloader
        train_dataset = self.train_data.unroll(metaconfig)
        train_data_loader_kwargs = {'num_workers': self.dataloader_workers, 
                                    'batch_size': self.bsize, 
                                    'collate_fn': train_dataset.collate}
        if not isinstance(train_dataset, IterableDataset):
            train_data_loader_kwargs['shuffle'] = True
        train_dataloader = DataLoader(train_dataset, **train_data_loader_kwargs)

        # setup training objects
        training_state, model, model_state, rng_keys = self.train_state.unroll(metaconfig)

        # define training step
        @partial(jax.jit, static_argnums=(0,), static_argnames=list(self.loss_kwargs.keys()))
        def step_fn(loss_fn, training_state, model_state, rngs, *args, **kwargs):
            def grad_loss(params, model_state, rngs, *args, **kwargs):
                variables = freeze({'params': params, **model_state})
                (loss, logs), variables = training_state.apply_fn(variables, *args, rngs=rngs, train=True, 
                                                                  mutable=True, method=loss_fn, **kwargs)
                model_state, _ = variables.pop('params')
                return loss, (logs, model_state)
            (_, (logs, model_state,)), grads = jax.value_and_grad(grad_loss, has_aux=True)(training_state.params, model_state, rngs, *args, **kwargs)
            training_state = training_state.apply_gradients(grads=grads)
            return logs, training_state, model_state

        # initalize training loop state
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])
        rng = self.rng.unroll(metaconfig)

        # train loop
        for epoch in tqdm(range(self.epochs)):
            for items in tqdm(train_dataloader):
                
                # step model and get training logs
                rng, new_rng = jax.random.split(rng)
                rngs = rngs_from_keys(new_rng, rng_keys)
                logs, training_state, model_state = step_fn(model.loss, training_state, model_state, rngs, *items, **self.loss_kwargs)
                train_logs.append(logs)
                
                # publish training logs
                if (training_state.step + 1) % self.log_every == 0:
                    logs = reduce_logs(train_logs)
                    logs = pool_logs(label_logs(logs, 'train', {'step': training_state.step, 'epoch': epoch}))
                    log(logs, self.use_wandb)
                
                # clear training logs
                if (training_state.step + 1) % self.train_state.optim.grad_accum_steps == 0:
                    train_logs = []
                
                # begin evaluation
                if (training_state.step + 1) % self.eval_every == 0:

                    # get eval logs
                    self.evaluator.model.variables = freeze({'params': training_state.params, **model_state})
                    eval_perf, eval_logs = self.evaluator.unroll(metaconfig)

                    # publish eval logs
                    eval_logs = pool_logs(label_logs(eval_logs, 'eval', {'step': training_state.step, 'epoch': epoch}))
                    log(eval_logs, self.use_wandb)

                    # conditionally save best model and optimizer state
                    if save_dir is not None and eval_perf < best_perf:
                        print('new best model! Saving ...')
                        with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
                            pkl.dump(to_bytes(freeze({'params': training_state.params, **model_state})), f)
                        with open(os.path.join(save_dir, 'optim.pkl'), 'wb') as f:
                            pkl.dump(to_bytes(training_state.opt_state), f)
                        print('saved.')
                        best_perf = eval_perf
                
                # periodically save checkpoint
                if save_dir is not None and self.save_every is not None and (training_state.step + 1) % self.save_every == 0:
                    print('saving checkpoint...')

                    # conditionally delete old checkpoints
                    if (self.max_steps is not None) and (len(saved_checkpoints) >= self.max_steps):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    
                    # save
                    with open(os.path.join(save_dir, 'model_%d.pkl' % (training_state.step)), 'wb') as f:
                        pkl.dump(to_bytes(freeze({'params': training_state.params, **model_state})), f)
                    saved_checkpoints.append(os.path.join(save_dir, 'model_%d.pkl' % (training_state.step)))
                    print('saved.')

                # conditionally terminate
                if self.max_steps is not None and training_state.step >= self.max_steps:
                    return
        
        # undo conditional jit block
        if not self.jit:
            fake_jit.stop()
