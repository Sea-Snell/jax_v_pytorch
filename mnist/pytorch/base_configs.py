from __future__ import annotations
from functools import partial
from typing import Callable, List
from micro_config import ConfigScript, MetaConfig
from torch_configs import ConfigScriptModel, ConfigScriptOptim
from dataclasses import dataclass
import mnist
from src import MLP, MNISTData, MNISTCNN
import os
import torch

project_root = os.path.dirname(__file__)

@dataclass
class MNISTDataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        if self.split == 'train':
            imgs = mnist.train_images()
            labels = mnist.train_labels()
        elif self.split == 'test':
            imgs = mnist.test_images()
            labels = mnist.test_labels()
        else:
            raise NotImplementedError
        return MNISTData(imgs=imgs, labels=labels)

@dataclass
class MLPConfig(ConfigScriptModel):
    shapes: List[int]
    dropout: float

    def unroll(self, metaconfig: MetaConfig) -> torch.nn.Module:
        return MLP(self.shapes, self.dropout)

@dataclass
class MNISTCNNConfig(ConfigScriptModel):
    def unroll(self, metaconfig: MetaConfig) -> torch.nn.Module:
        return MNISTCNN()

@dataclass
class AdamWConfig(ConfigScriptOptim):
    lr: float
    weight_decay: float

    def unroll(self, metaconfig: MetaConfig) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
        return lambda model: torch.optim.AdamW(model.parameters(), lr=self.lr, 
                                               weight_decay=self.weight_decay)
