from typing import Any, Iterator
import jax
import numpy as np
import collections
import itertools

def batch_idxs(rng: jax.random.KeyArray, data_size: int, bsize: int) -> np.ndarray:
    steps_per_epoch = data_size // bsize
    permutations = np.asarray(jax.random.permutation(rng, data_size))
    permutations = permutations[:steps_per_epoch * bsize]
    permutations = permutations.reshape(steps_per_epoch, bsize)
    return permutations

def batch_iterator(rng: jax.random.KeyArray, dataset: Any, bsize: int) -> Iterator:
    batches = batch_idxs(rng, len(dataset), bsize)
    for idxs in batches:
        yield dataset[idxs]

def prefetch(iterator: Iterator, queue_size: int = 2) -> Iterator:
    # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    # queue_size = 2 should be ok for one GPU.

    queue = collections.deque()

    def enqueue(n):
        for item in itertools.islice(iterator, n):
            queue.append(item)

    enqueue(queue_size)
    while queue:
        yield queue.popleft()
        enqueue(1)
