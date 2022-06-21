from __future__ import annotations
from functools import partial
from typing import List, Tuple, Union
from micro_config import ConfigScript, MetaConfig
from haiku_configs import ConfigScriptModel, ConfigScriptOptim, ModelConfigReturn
from dataclasses import dataclass
import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax
from src import MLP, SimpleCNN, ImageData
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
    shapes: List[int]
    dropout: float

    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = hk.multi_transform_with_state(partial(MLP.multi_transform_f, self.shapes[1:], self.dropout))
        return ModelConfigReturn(model, (jnp.zeros((1, self.shapes[0],)),), {'train': True})

@dataclass
class SimpleCNNConfig(ConfigScriptModel):
    img_shape: Union[List[int], Tuple[int]]
    n_labels: int

    def unroll(self, metaconfig: MetaConfig) -> ModelConfigReturn:
        model = hk.multi_transform_with_state(partial(SimpleCNN.multi_transform_f, self.n_labels))
        return ModelConfigReturn(model, (jnp.zeros((1, *self.img_shape,)),), {'train': True})

@dataclass
class AdamWConfig(ConfigScriptOptim):
    lr: float
    weight_decay: float

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        return optax.adamw(self.lr, weight_decay=self.weight_decay)
