from __future__ import annotations
from collections import namedtuple
from typing import Any, Callable, Optional, Union, ClassVar
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax
import pickle as pkl
import optax
from flax_utils import rngs_from_keys
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
from abc import abstractmethod

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

ModelConfigReturn = namedtuple('ModelConfigReturn', ['model', 'rng_keys', 'init_args', 'init_kwargs'])
ModelConfigResult = namedtuple('ModelConfigResult', ['model', 'variables', 'rng_keys'])

@dataclass
class ConfigScriptModel(ConfigScript):
    rng: ConfigScriptRNG
    checkpoint_path: Optional[str]
    variables: Optional[Any]

    def meta_unroll(unroll: Callable[[MetaConfig], ModelConfigReturn]):
        def new_unroll(self, metaconfig: MetaConfig) -> ModelConfigResult:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                result = metaconfig.unrolled[id(self)]
                if self.variables is not None:
                    result = result._replace(variables=self.variables)
                return result
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            temp_result = unroll(self, metaconfig)
            rng = self.rng.unroll(metaconfig)
            rngs = rngs_from_keys(rng, {'params', *temp_result.rng_keys})
            variables = temp_result.model.init(rngs, *temp_result.init_args, **temp_result.init_kwargs)
            if self.variables is not None:
                variables = self.variables
            elif self.checkpoint_path is not None:
                if metaconfig.verbose:
                    print('loading model state from: %s' % metaconfig.convert_path(self.checkpoint_path))
                with open(metaconfig.convert_path(self.checkpoint_path), 'rb') as f:
                    variables = from_bytes(variables, pkl.load(f))
                if metaconfig.verbose:
                    print('loaded.')
            result = ModelConfigResult(temp_result.model, variables, temp_result.rng_keys)
            
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
            _, variables, _ = self.model.unroll(metaconfig)
            _, params = variables.pop('params')
            optim_state = optimizer.init(params)
            if self.optim_state is not None:
                optim_state = self.optim_state
            elif self.state_path is not None:
                if metaconfig.verbose:
                    print('loading optimizer state from: %s' % metaconfig.convert_path(self.state_path))
                with open(metaconfig.convert_path(self.state_path), 'rb') as f:
                    optim_state = from_bytes(optim_state, pkl.load(f))
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

TrainStateConfigResult = namedtuple('TrainStateConfigResult', ['train_state', 'model', 'model_state', 'rng_keys'])

@dataclass
class TrainStateConfig(ConfigScript):
    model: ConfigScriptModel
    optim: ConfigScriptOptim
    step: int=0

    def unroll(self, metaconfig: MetaConfig) -> TrainStateConfigResult:
        model, variables, rng_keys = self.model.unroll(metaconfig)
        model_state, params = variables.pop('params')
        optimizer, optimizer_state = self.optim.unroll(metaconfig)
        train_state = TrainState(
                                    self.step, 
                                    model.apply, 
                                    params, 
                                    optimizer, 
                                    optimizer_state, 
                                )
        return TrainStateConfigResult(train_state, model, model_state, rng_keys)
