from __future__ import annotations
from collections import namedtuple
from typing import Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax

@dataclass
class RNGSeed(ConfigScript):
    value: int

    def unroll(self, metaconfig: MetaConfig) -> jax.random.KeyArray:
        return jax.random.PRNGKey(self.value)
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)

@dataclass
class RNGSplit(ConfigScript):
    seed: RNGSeed
    n_splits: int

    def unroll(self, metaconfig: MetaConfig) -> jax.random.KeyArray:
        rng = self.seed.unroll(metaconfig)
        if self.n_splits == 0:
            return rng
        for _ in range(self.n_splits):
            rng, new_rng = jax.random.split(rng)
        return new_rng
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)

ConfigScriptRNG = Union[RNGSeed, RNGSplit]
