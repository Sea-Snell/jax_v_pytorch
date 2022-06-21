from typing import List, Union
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from logs import LogTuple

class MNISTData:
    def __init__(self, imgs, labels):
        self.imgs = imgs.reshape(-1, 28*28)
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx: Union[int, np.ndarray]):
        imgs, labels = self.imgs[idx], self.labels[idx]
        imgs, labels = jnp.uint8(imgs), jnp.uint8(labels)
        labels = jax.nn.one_hot(labels, 10, dtype=jnp.uint8)
        return imgs, labels

class MLP(nn.Module):
    output_shapes: List[int]
    dropout_rate: float
    
    def setup(self):
        self.linears = [nn.Dense(self.output_shapes[i]) for i in range(len(self.output_shapes))]
        self.dropouts = [nn.Dropout(self.dropout_rate) for _ in range(len(self.output_shapes)-1)]
    
    def __call__(self, x, *, train):
        x = x.astype(jnp.float32) / 256.0
        for i in range(len(self.output_shapes)-1):
            x = self.linears[i](x)
            x = self.dropouts[i](x, deterministic=not train)
            x = jax.nn.relu(x)
        x = self.linears[-1](x)
        return x
    
    def loss(self, x, y, *, train):
        y = y.astype(jnp.float32)
        n = y.shape[0]
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, logs

class MNISTCNN(nn.Module):
    def setup(self):
        self.block1 = nn.Sequential([
            nn.Conv(32, (5, 5), strides=1, padding='SAME'), 
            jax.nn.relu, 
            nn.Conv(32, (5, 5), strides=1, padding='SAME'), 
            jax.nn.relu
        ])
        self.dropout1 = nn.Dropout(0.25)
        self.block2 = nn.Sequential([
            nn.Conv(64, (3, 3), strides=1, padding='SAME'), 
            jax.nn.relu, 
            nn.Conv(64, (3, 3), strides=1, padding='SAME'), 
            jax.nn.relu
        ])
        self.dropout2 = nn.Dropout(0.25)
        self.linear1 = nn.Dense(128)
        self.dropout3 = nn.Dropout(0.5)
        self.linear2 = nn.Dense(10)
    
    def __call__(self, x, *, train):
        x = x.astype(jnp.float32) / 256.0
        x = jnp.reshape(x, (-1, 28, 28, 1))
        x = self.block1(x)
        x = nn.max_pool(x, (2, 2), (2, 2), padding='VALID')
        x = self.dropout1(x, deterministic=not train)
        x = self.block2(x)
        x = nn.max_pool(x, (2, 2), (2, 2), padding='VALID')
        x = self.dropout2(x, deterministic=not train)
        x = jnp.reshape(x, (-1, 7*7*64))
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout3(x, deterministic=not train)
        x = self.linear2(x)
        return x
    
    def loss(self, x, y, *, train):
        y = y.astype(jnp.float32)
        n = y.shape[0]
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, logs
