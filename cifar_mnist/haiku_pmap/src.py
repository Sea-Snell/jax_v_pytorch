from collections import namedtuple
from typing import List, Union, Tuple
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from jaxtyping import PyTree, f32, u8

class ImageData:
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, n_labels: int):
        self.imgs = imgs
        self.labels = labels
        self.n_labels = n_labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Tuple[u8, u8]:
        imgs, labels = self.imgs[idx], self.labels[idx]
        imgs, labels = jnp.uint8(imgs), jnp.uint8(labels)
        return imgs, labels

def random_crop_augment(rng: jax.random.KeyArray, img: f32["h w d"], padding: int) -> f32["h w d"]:
    crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate((crop_from, jnp.zeros((1,), dtype=jnp.int32),), axis=0)
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0),), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

def random_flip_augment(rng: jax.random.KeyArray, img: f32["h w d"]) -> f32["h w d"]:
    should_flip = jax.random.bernoulli(rng, 0.5)
    return jax.lax.cond(should_flip, lambda x: jnp.flip(x, axis=1), lambda x: x, img)

def batched_augmentation(rng: jax.random.KeyArray, imgs: f32["b h w d"], padding: int=4) -> f32["b h w d"]:
    rng, *new_rngs = jax.random.split(rng, imgs.shape[0]+1)
    imgs = jax.vmap(random_crop_augment, (0, 0, None,))(jnp.stack(new_rngs, axis=0), imgs, padding)
    rng, *new_rngs = jax.random.split(rng, imgs.shape[0]+1)
    imgs = jax.vmap(random_flip_augment, (0, 0,))(jnp.stack(new_rngs, axis=0), imgs)
    return imgs

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
    
    def __call__(self, x: u8['b h w d'], *, train: bool, do_aug: bool=True, crop_aug_padding: int=4) -> f32['b label']:
        dropout_rate = self.dropout_rate if train else 0.0
        x = x.astype(jnp.float32) / 255.0
        if train and do_aug:
            x = batched_augmentation(hk.next_rng_key(), x, crop_aug_padding)
        x = hk.Flatten()(x)
        for output_shape in self.output_shapes[:-1]:
            x = hk.Linear(output_shape)(x)
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            x = jax.nn.relu(x)
        x = hk.Linear(self.output_shapes[-1])(x)
        return x
    
    def loss(self, x: u8['b h w d'], y: u8['b'], *, train: bool, do_aug: bool=True, crop_aug_padding: int=4) -> Tuple[f32, PyTree]:
        predictions = self(x, train=train, do_aug=do_aug, crop_aug_padding=crop_aug_padding)
        loss = optax.softmax_cross_entropy_with_integer_labels(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == y).mean()}
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
    
    def __call__(self, x: u8['b h w d'], *, train: bool, do_aug: bool=True, crop_aug_padding: int=4) -> f32['b label']:
        dropout_rate1 = 0.25 if train else 0.0
        dropout_rate2 = 0.5 if train else 0.0

        x = x.astype(jnp.float32) / 255.0
        if train and do_aug:
            x = batched_augmentation(hk.next_rng_key(), x, crop_aug_padding)
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
    
    def loss(self, x: u8['b h w d'], y: u8['b'], *, train: bool, do_aug: bool=True, crop_aug_padding: int=4) -> Tuple[f32, PyTree]:
        predictions = self(x, train=train, do_aug=do_aug, crop_aug_padding=crop_aug_padding)
        loss = optax.softmax_cross_entropy_with_integer_labels(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == y).mean()}
        return loss, logs
