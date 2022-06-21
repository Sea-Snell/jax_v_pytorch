from __future__ import annotations
from functools import partial
from typing import Callable, List
from micro_config import ConfigScript, MetaConfig
from torch_configs import ConfigScriptModel, ConfigScriptOptim
from dataclasses import dataclass
from src import MLP, SimpleCNN, ImageData
import os
import torch
import numpy as np
import torchvision.datasets as datasets

project_root = os.path.dirname(__file__)

@dataclass
class MNISTDataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        data = datasets.MNIST(root='../data/', train=(self.split == 'train'), download=True)
        imgs, labels = list(zip(*map(lambda x: map(np.array, x), data)))
        imgs, labels = np.stack(imgs, axis=0), np.stack(labels, axis=0)
        imgs = np.expand_dims(imgs, axis=1)
        return ImageData(imgs=imgs, labels=labels, n_labels=10)

@dataclass
class FashionMNISTDataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        data = datasets.FashionMNIST(root='../data/', train=(self.split == 'train'), download=True)
        imgs, labels = list(zip(*map(lambda x: map(np.array, x), data)))
        imgs, labels = np.stack(imgs, axis=0), np.stack(labels, axis=0)
        imgs = np.expand_dims(imgs, axis=1)
        return ImageData(imgs=imgs, labels=labels, n_labels=10)

@dataclass
class CIFAR10DataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        data = datasets.CIFAR10(root='../data/', train=(self.split == 'train'), download=True)
        imgs, labels = list(zip(*map(lambda x: map(np.array, x), data)))
        imgs, labels = np.stack(imgs, axis=0), np.stack(labels, axis=0)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        return ImageData(imgs=imgs, labels=labels, n_labels=10)

@dataclass
class CIFAR100DataConfig(ConfigScript):
    split: str

    def __post_init__(self):
        assert self.split in {'train', 'test'}

    def unroll(self, metaconfig: MetaConfig):
        data = datasets.CIFAR100(root='../data/', train=(self.split == 'train'), download=True)
        imgs, labels = list(zip(*map(lambda x: map(np.array, x), data)))
        imgs, labels = np.stack(imgs, axis=0), np.stack(labels, axis=0)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        return ImageData(imgs=imgs, labels=labels, n_labels=100)

@dataclass
class MLPConfig(ConfigScriptModel):
    shapes: List[int]
    dropout: float

    def unroll(self, metaconfig: MetaConfig) -> torch.nn.Module:
        return MLP(self.shapes, self.dropout)

@dataclass
class SimpleCNNConfig(ConfigScriptModel):
    img_size: int
    n_channels: int
    n_labels: int

    def unroll(self, metaconfig: MetaConfig) -> torch.nn.Module:
        return SimpleCNN(self.img_size, self.n_channels, self.n_labels)

@dataclass
class AdamWConfig(ConfigScriptOptim):
    lr: float
    weight_decay: float

    def unroll(self, metaconfig: MetaConfig) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
        return lambda model: torch.optim.AdamW(model.parameters(), lr=self.lr, 
                                               weight_decay=self.weight_decay)
