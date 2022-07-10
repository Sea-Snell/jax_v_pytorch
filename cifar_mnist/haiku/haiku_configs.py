from __future__ import annotations
from abc import abstractmethod
from collections import namedtuple
from typing import Any, Callable, Optional, Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax
import pickle as pkl
import optax

@dataclass
class RNGSeed(ConfigScript):
    value: int

    def unroll(self, metaconfig: MetaConfig) -> jax.random.KeyArray:
        return jax.random.PRNGKey(self.value)
    
    def split(self, n_splits: int) -> RNGSplit:
        return RNGSplit(self, n_splits)
    
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

ConfigScriptRNG = Union[RNGSeed, RNGSplit]

ModelConfigReturn = namedtuple('ModelConfigReturn', ['model', 'init_args', 'init_kwargs'])
ModelConfigResult = namedtuple('ModelConfigResult', ['model', 'params', 'state'])

@dataclass
class ConfigScriptModel(ConfigScript):
    rng: ConfigScriptRNG
    checkpoint_path: Optional[str]
    params: Optional[Any]
    state: Optional[Any]

    def __post_init__(self):
        assert (self.params is not None and self.state is not None) or (self.params is None and self.state is None)

    def meta_unroll(unroll: Callable[[MetaConfig], ModelConfigReturn]):
        def new_unroll(self, metaconfig: MetaConfig) -> ModelConfigResult:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                result = metaconfig.unrolled[id(self)]
                if self.params is not None:
                    result = result._replace(params=self.params, state=self.state)
                return result
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            temp_result = unroll(self, metaconfig)
            rng = self.rng.unroll(metaconfig)
            params, state = temp_result.model.init(rng, *temp_result.init_args, **temp_result.init_kwargs)
            if self.params is not None:
                params, state = self.params, self.state
            elif self.checkpoint_path is not None:
                if metaconfig.verbose:
                    print('loading model state from: %s' % metaconfig.convert_path(self.checkpoint_path))
                with open(metaconfig.convert_path(self.checkpoint_path), 'rb') as f:
                    params, state = pkl.load(f)
                if metaconfig.verbose:
                    print('loaded.')
            result = ModelConfigResult(temp_result.model, params, state)
            
            metaconfig.unrolled[id(self)] = result
            if metaconfig.verbose:
                print(f'unrolled {self.__class__.__name__} and cached: {id(self)}')
            return result
        return new_unroll
    
    @abstractmethod
    def unroll(metaconfig: MetaConfig) -> ModelConfigReturn:
        pass

OptimConfigResult = namedtuple('OptimConfigResult', ['optim', 'optim_state'])

@dataclass
class ConfigScriptOptim(ConfigScript):
    grad_accum_steps: int
    model: ConfigScriptModel
    state_path: Optional[str]
    optim_state: Optional[Any]

    def meta_unroll(unroll: Callable[[MetaConfig], optax.GradientTransformation]):
        def new_unroll(self, metaconfig: MetaConfig) -> OptimConfigResult:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                result = metaconfig.unrolled[id(self)]
                if self.optim_state is not None:
                    result = result._replace(optim_state=self.optim_state)
                return result
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            optimizer = unroll(self, metaconfig)
            optimizer = optax.MultiSteps(optimizer, 
                                         self.grad_accum_steps, 
                                         use_grad_mean=True)
            _, params, _ = self.model.unroll(metaconfig)
            optim_state = optimizer.init(params)
            if self.optim_state is not None:
                optim_state = self.optim_state
            elif self.state_path is not None:
                if metaconfig.verbose:
                    print('loading optimizer state from: %s' % metaconfig.convert_path(self.state_path))
                with open(metaconfig.convert_path(self.state_path), 'rb') as f:
                    optim_state = pkl.load(f)
                if metaconfig.verbose:
                    print('loaded.')
            result = OptimConfigResult(optimizer, optim_state,)

            metaconfig.unrolled[id(self)] = result
            if metaconfig.verbose:
                print(f'unrolled {self.__class__.__name__} and cached: {id(self)}')
            return result
        return new_unroll
    
    @abstractmethod
    def unroll(metaconfig: MetaConfig) -> optax.GradientTransformation:
        pass
