from __future__ import annotations
from typing import List, Tuple, Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import optax
from src import MLP, SimpleCNN, ImageData
from flax_configs import ModelConfigReturn, ConfigScriptModel, ConfigScriptOptim
import os
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
        imgs = np.expand_dims(imgs, axis=3)
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
        imgs = np.expand_dims(imgs, axis=3)
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
        return ImageData(imgs=imgs, labels=labels, n_labels=100)

@dataclass
class MLPConfig(ConfigScriptModel):
    img_shape: Union[List[int], Tuple[int]]
    out_shapes: List[int]
    dropout: float
    
    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = MLP(self.out_shapes, self.dropout)
        return ModelConfigReturn(model, frozenset({'dropout', 'augment'}), (jnp.zeros((1, *self.img_shape,)),), {'train': True})

@dataclass
class SimpleCNNConfig(ConfigScriptModel):
    img_shape: Union[List[int], Tuple[int]]
    n_labels: int

    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = SimpleCNN(self.n_labels)
        return ModelConfigReturn(model, frozenset({'dropout', 'augment'}), (jnp.zeros((1, *self.img_shape,)),), {'train': True})

@dataclass
class AdamWConfig(ConfigScriptOptim):
    lr: float
    weight_decay: float

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        return optax.adamw(self.lr, weight_decay=self.weight_decay)
