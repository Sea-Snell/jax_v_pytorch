from collections import namedtuple
from typing import List
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from torch.utils.data import Dataset
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
        imgs, labels = jnp.reshape(imgs, (-1, 28*28)).astype(jnp.float32), jax.nn.one_hot(labels, 10)
        return imgs, labels

MLP_transformed = namedtuple('MLP_transformed', ['forward', 'loss'])

class MLP(hk.Module):
    def __init__(self, output_shapes: List[int], dropout_rate: float):
        super().__init__(name='mlp')
        self.dropout_rate = dropout_rate
        self.output_shapes = output_shapes
    
    @classmethod
    def multi_transform_f(cls, *args, **kwargs):
        model = cls(*args, **kwargs)
        return model.__call__, MLP_transformed(model.__call__, model.loss)
    
    def __call__(self, x, *, train):
        dropout_rate = self.dropout_rate if train else 0.0
        for output_shape in self.output_shapes[:-1]:
            x = hk.Linear(output_shape)(x)
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            x = jax.nn.relu(x)
        x = hk.Linear(self.output_shapes[-1])(x)
        return x
    
    def loss(self, x, y, *, train):
        n = y.shape[0]
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, logs

MNISTCNN_transformed = namedtuple('MNISTCNN_transformed', ['forward', 'loss'])

class MNISTCNN(hk.Module):
    def __init__(self):
        super().__init__(name='mnist_cnn')
    
    @classmethod
    def multi_transform_f(cls, *args, **kwargs):
        model = cls(*args, **kwargs)
        return model.__call__, MLP_transformed(model.__call__, model.loss)
    
    def __call__(self, x, *, train):
        dropout_rate1 = 0.25 if train else 0.0
        dropout_rate2 = 0.5 if train else 0.0

        x = hk.Reshape((28, 28, 1))(x)
        x = hk.Sequential([
                hk.Conv2D(32, kernel_shape=(5, 5), stride=1, padding='SAME'), 
                jax.nn.relu, 
                hk.Conv2D(32, kernel_shape=(5, 5), stride=1, padding='SAME'), 
                jax.nn.relu, 
            ])(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = hk.dropout(hk.next_rng_key(), dropout_rate1, x)
        x = hk.Sequential([
            hk.Conv2D(64, kernel_shape=(3, 3), stride=1, padding='SAME'), 
            jax.nn.relu, 
            hk.Conv2D(64, kernel_shape=(3, 3), stride=1, padding='SAME'), 
            jax.nn.relu,  
        ])(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = hk.dropout(hk.next_rng_key(), dropout_rate1, x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate2, x)
        x = hk.Linear(10)(x)
        return x
    
    def loss(self, x, y, *, train):
        n = y.shape[0]
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean(), n)}
        return loss, logs
