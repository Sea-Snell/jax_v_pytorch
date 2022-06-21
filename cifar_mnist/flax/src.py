from typing import List, Union
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

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

def random_crop_augment(rng: jax.random.KeyArray, img: jnp.ndarray, padding: int) -> jnp.ndarray:
    crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate((crop_from, jnp.zeros((1,), dtype=jnp.int32),), axis=0)
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0),), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

def random_flip_augment(rng: jax.random.KeyArray, img: jnp.ndarray) -> jnp.ndarray:
    should_flip = jax.random.bernoulli(rng, 0.5)
    return jax.lax.cond(should_flip, lambda x: jnp.flip(x, axis=1), lambda x: x, img)

def batched_augmentation(rng: jax.random.KeyArray, imgs: jnp.ndarray, padding: int=4):
    rng, *new_rngs = jax.random.split(rng, imgs.shape[0]+1)
    imgs = jax.vmap(random_crop_augment, (0, 0, None,))(jnp.stack(new_rngs, axis=0), imgs, padding)
    rng, *new_rngs = jax.random.split(rng, imgs.shape[0]+1)
    imgs = jax.vmap(random_flip_augment, (0, 0,))(jnp.stack(new_rngs, axis=0), imgs)
    return imgs

class MLP(nn.Module):
    output_shapes: List[int]
    dropout_rate: float
    
    def setup(self):
        self.linears = [nn.Dense(self.output_shapes[i]) for i in range(len(self.output_shapes))]
        self.dropouts = [nn.Dropout(self.dropout_rate) for _ in range(len(self.output_shapes)-1)]
    
    def __call__(self, x, *, train: bool, do_aug: bool=True, crop_aug_padding: int=4):
        x = x.astype(jnp.float32) / 255.0
        if train and do_aug:
            x = batched_augmentation(self.make_rng('augment'), x, crop_aug_padding)
        x = jnp.reshape(x, (x.shape[0], -1))
        for i in range(len(self.output_shapes)-1):
            x = self.linears[i](x)
            x = self.dropouts[i](x, deterministic=not train)
            x = jax.nn.relu(x)
        x = self.linears[-1](x)
        return x
    
    def loss(self, x, y, *, train: bool, do_aug: bool=True, crop_aug_padding: int=4):
        y = y.astype(jnp.float32)
        predictions = self(x, train=train, do_aug=do_aug, crop_aug_padding=crop_aug_padding)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean()}
        return loss, logs

class SimpleCNN(nn.Module):
    n_labels: int

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
        self.linear2 = nn.Dense(self.n_labels)
    
    def __call__(self, x, *, train: bool, do_aug: bool=True, crop_aug_padding: int=4):
        x = x.astype(jnp.float32) / 255.0
        if train and do_aug:
            x = batched_augmentation(self.make_rng('augment'), x, crop_aug_padding)
        x = self.block1(x)
        x = nn.max_pool(x, (2, 2), (2, 2), padding='VALID')
        x = self.dropout1(x, deterministic=not train)
        x = self.block2(x)
        x = nn.max_pool(x, (2, 2), (2, 2), padding='VALID')
        x = self.dropout2(x, deterministic=not train)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout3(x, deterministic=not train)
        x = self.linear2(x)
        return x
    
    def loss(self, x, y, *, train: bool, do_aug: bool=True, crop_aug_padding: int=4):
        y = y.astype(jnp.float32)
        predictions = self(x, train=train, do_aug=do_aug, crop_aug_padding=crop_aug_padding)
        loss = optax.softmax_cross_entropy(predictions, y).mean()
        logs = {'loss': loss, 'acc': (jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1)).mean()}
        return loss, logs
