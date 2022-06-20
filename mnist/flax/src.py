from collections import namedtuple
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import Dataset
from flax.core import freeze, unfreeze
from flax import linen as nn
from logs import LogTuple

class MNISTData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    @staticmethod
    def collate(items):
        imgs, labels = zip(*items)
        imgs, labels = jnp.array(np.stack(imgs, axis=0)), jnp.array(np.stack(labels, axis=0))
        imgs, labels = jnp.reshape(imgs, (-1, 28*28)), jax.nn.one_hot(labels, 10)
        return imgs, labels

class MLP(nn.Module):
    output_shapes: List[int]
    dropout_rate: float
    
    def setup(self):
        self.linears = [nn.Dense(self.output_shapes[i]) for i in range(len(self.output_shapes))]
        self.dropouts = [nn.Dropout(self.dropout_rate) for _ in range(len(self.output_shapes)-1)]
    
    def __call__(self, x, *, train):
        for i in range(len(self.output_shapes)-1):
            x = self.linears[i](x)
            x = self.dropouts[i](x, deterministic=not train)
            x = jax.nn.relu(x)
        x = self.linears[-1](x)
        return x
    
    def loss(self, x, y, *, train):
        n = y.shape[0]
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, logs

