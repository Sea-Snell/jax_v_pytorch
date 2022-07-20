from micro_config import ConfigScript, MetaConfig
from collections import namedtuple
from dataclasses import dataclass
from abc import abstractmethod
import jax.numpy as jnp
import jax

HFPjitModelResult = namedtuple("HFPjitModelResult", ["model", "params", "tokenizer", "rules"])

@dataclass
class PretrainedHFPjitModelConfig(ConfigScript):
    model_str: str
    use_fp16: bool

    def get_dtype(self):
        if self.use_fp16:
            if jax.default_backend() == 'tpu':
                return jnp.bfloat16
            return jnp.float16
        return jnp.float32
    
    def params_to_dtype(self, model, params):
        dtype = self.get_dtype()
        if dtype == jnp.bfloat16:
            return model.to_bf16(params)
        elif dtype == jnp.float16:
            return model.to_fp16(params)
        return model.to_fp32(params)

    @abstractmethod
    def unroll(self, metaconfig: MetaConfig) -> HFPjitModelResult:
        pass
