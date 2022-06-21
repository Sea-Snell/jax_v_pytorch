from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
import torch
from typing import Callable, Optional, Union
from micro_config import ConfigScript, MetaConfig

@dataclass
class DeviceConfig(ConfigScript):
    device_str: str

    @classmethod
    def gpu_if_available(cls) -> DeviceConfig:
        return cls('cuda' if torch.cuda.is_available() else 'cpu')

    def unroll(self, metaconfig: MetaConfig) -> torch.device:
        return torch.device(self.device_str)

@dataclass
class ConfigScriptModel(ConfigScript):
    checkpoint_path: Optional[str]
    strict_load: bool
    device: DeviceConfig

    def meta_unroll(unroll):
        def new_unroll(self, metaconfig: MetaConfig) -> torch.nn.Module:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                return metaconfig.unrolled[id(self)]
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            model = unroll(self, metaconfig)
            device = self.device.unroll(metaconfig)
            model = model.to(device)
            if self.checkpoint_path is not None:
                if metaconfig.verbose:
                    print('loading state dict from: %s' % metaconfig.convert_path(self.checkpoint_path))
                model.load_state_dict(torch.load(metaconfig.convert_path(self.checkpoint_path), map_location='cpu'), strict=self.strict_load)
                if metaconfig.verbose:
                    print('loaded.')
            
            metaconfig.unrolled[id(self)] = model
            if metaconfig.verbose:
                print(f'unrolled and {self.__class__.__name__} cached: {id(self)}')
            return model
        return new_unroll
    
    @abstractmethod
    def unroll(metaconfig: MetaConfig) -> torch.nn.Module:
        pass

@dataclass
class ConfigScriptOptim(ConfigScript):
    model: ConfigScriptModel
    state_path: Optional[str]

    def meta_unroll(unroll: Callable[[MetaConfig], Callable[[torch.nn.Module], torch.optim.Optimizer]]):
        def new_unroll(self, metaconfig: MetaConfig) -> torch.optim.Optimizer:
            if metaconfig.unrolled is None:
                metaconfig.unrolled = {}
            if id(self) in metaconfig.unrolled:
                if metaconfig.verbose:
                    print(f'fetching {self.__class__.__name__} from cache: {id(self)}')
                return metaconfig.unrolled[id(self)]
            if metaconfig.verbose:
                print(f'unrolling {self.__class__.__name__}: {id(self)}')
            
            optim_f = unroll(self, metaconfig)
            model = self.model.unroll(metaconfig)
            optim = optim_f(model)
            if self.state_path is not None:
                if metaconfig.verbose:
                    print('loading state dict from: %s' % metaconfig.convert_path(self.state_path))
                optim.load_state_dict(torch.load(metaconfig.convert_path(self.state_path), map_location='cpu'))
                if metaconfig.verbose:
                    print('loaded.')
            
            metaconfig.unrolled[id(self)] = optim
            if metaconfig.verbose:
                print(f'unrolled and {self.__class__.__name__} cached: {id(self)}')
            return optim
        return new_unroll
    
    @abstractmethod
    def unroll(metaconfig: MetaConfig) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
        pass
