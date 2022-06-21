import jax
import numpy as np

def batch_idxs(rng, data_size, bsize):
    steps_per_epoch = data_size // bsize
    permutations = np.asarray(jax.random.permutation(rng, data_size))
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    return permutations
