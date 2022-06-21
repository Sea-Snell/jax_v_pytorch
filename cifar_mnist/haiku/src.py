from collections import namedtuple
from typing import List, Union
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

class ImageData:
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, n_labels: int):
        self.imgs = imgs
        self.labels = labels
        self.n_labels = n_labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx: Union[int, np.ndarray]):
        imgs, labels = self.imgs[idx], self.labels[idx]
        imgs, labels = jnp.uint8(imgs), jnp.uint8(labels)
        labels = jax.nn.one_hot(labels, self.n_labels, dtype=jnp.uint8)
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
        
        x = x.astype(jnp.float32) / 255.0
        x = hk.Flatten()(x)
        for output_shape in self.output_shapes[:-1]:
            x = hk.Linear(output_shape)(x)
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            x = jax.nn.relu(x)
        x = hk.Linear(self.output_shapes[-1])(x)
        return x
    
    def loss(self, x, y, *, train):
        y = y.astype(jnp.float32)
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean()}
        return loss, logs

SimpleCNN_transformed = namedtuple('SimpleCNN_transformed', ['forward', 'loss'])

class SimpleCNN(hk.Module):
    def __init__(self, n_labels: int):
        super().__init__(name='simple_cnn')
        self.n_labels = n_labels
    
    @classmethod
    def multi_transform_f(cls, *args, **kwargs):
        model = cls(*args, **kwargs)
        return model.__call__, SimpleCNN_transformed(model.__call__, model.loss)
    
    def __call__(self, x, *, train):
        dropout_rate1 = 0.25 if train else 0.0
        dropout_rate2 = 0.5 if train else 0.0

        x = x.astype(jnp.float32) / 255.0
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
        x = hk.Linear(self.n_labels)(x)
        return x
    
    def loss(self, x, y, *, train):
        y = y.astype(jnp.float32)
        predictions = self(x, train=train)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean()}
        return loss, logs
