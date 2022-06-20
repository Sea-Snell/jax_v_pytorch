from typing import Optional, Dict, Any
from micro_config import ConfigScript, ConfigScriptNoCache, MetaConfig
from dataclasses import dataclass, asdict
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from collections import deque
import os
import pickle as pkl
from logs import label_logs, pool_logs, reduce_logs, log
from tqdm.auto import tqdm
import wandb
from torch_utils import to
from torch_configs import ConfigScriptModel, ConfigScriptOptim
import tree
import torch
import json

@dataclass
class StandardaEvaluator(ConfigScriptNoCache):
    eval_data: ConfigScript
    model: ConfigScriptModel
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
        model = self.model.unroll(metaconfig)
        device = self.model.device.unroll(metaconfig)

        # setup evaluator loop state
        eval_logs = []

        # eval on batches
        for i, items in tqdm(enumerate(eval_dataloader)):
            
            # conditionally terminate early
            if self.eval_batches is not None and i >= self.eval_batches:
                break

            # get eval logs
            items = to(tree.map_structure(lambda x: torch.tensor(x), items), device)
            _, logs = model.loss(*items, **self.loss_kwargs)
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
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    grad_accum_steps: int
    log_every: int
    eval_every: int
    save_every: Optional[int]
    dataloader_workers: int
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
        
        # setup dataloader
        train_dataset = self.train_data.unroll(metaconfig)
        train_data_loader_kwargs = {'num_workers': self.dataloader_workers, 
                                    'batch_size': self.bsize, 
                                    'collate_fn': train_dataset.collate}
        if not isinstance(train_dataset, IterableDataset):
            train_data_loader_kwargs['shuffle'] = True
        train_dataloader = DataLoader(train_dataset, **train_data_loader_kwargs)

        # setup training objects
        model = self.model.unroll(metaconfig)
        optim = self.optim.unroll(metaconfig)
        device = self.model.device.unroll(metaconfig)

        # initalize training loop state
        step = 0
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])
        model.train()

         # train loop
        for epoch in tqdm(range(self.epochs)):
            for items in tqdm(train_dataloader):
                
                # accumulate loss gradients and save training logs
                items = to(tree.map_structure(lambda x: torch.tensor(x), items), device)
                loss, logs = model.loss(*items, **self.loss_kwargs)
                (loss / self.grad_accum_steps).backward()
                train_logs.append(logs)
                
                # step accumulated gradients
                if (step + 1) % self.grad_accum_steps == 0:
                    optim.step()
                    optim.zero_grad()
                
                # publish training logs
                if (step + 1) % self.log_every == 0:
                    logs = reduce_logs(train_logs)
                    logs = pool_logs(label_logs(logs, 'train', {'step': step, 'epoch': epoch}))
                    log(logs, self.use_wandb)
                
                # clear training logs
                if (step + 1) % self.grad_accum_steps == 0:
                    train_logs = []
                
                # begin evaluation
                if (step + 1) % self.eval_every == 0:
                    
                    # set model to eval mode
                    model.eval()

                    # get eval logs
                    eval_perf, eval_logs = self.evaluator.unroll(metaconfig)

                    # publish eval logs
                    eval_logs = pool_logs(label_logs(eval_logs, 'eval', {'step': step, 'epoch': epoch}))
                    log(eval_logs, self.use_wandb)
                    
                    # conditionally save best model and optimizer state
                    if save_dir is not None and eval_perf < best_perf:
                        print('new best eval loss! Saving ...')
                        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pkl'))
                        torch.save(optim.state_dict(), os.path.join(save_dir, 'optim.pkl'))
                        print('saved.')
                        best_perf = eval_perf
                    
                    # reset model to training mode
                    model.train()
                
                # periodically save checkpoint
                if save_dir is not None and self.save_every is not None and (step + 1) % self.save_every == 0:
                    print('saving checkpoint...')
                    
                    # conditionally delete old checkpoints
                    if (self.max_checkpoints is not None) and (len(saved_checkpoints) >= self.max_checkpoints):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    
                    # save
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model_%d.pkl' % (step)))
                    saved_checkpoints.append(os.path.join(save_dir, 'model_%d.pkl' % (step)))
                    print('saved.')
                
                # increment step counter
                step += 1
                
                # conditionally terminate
                if self.max_steps is not None and step >= self.max_steps:
                    return

