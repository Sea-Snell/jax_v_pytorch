import jax
from transformers import FlaxGPTJForCausalLM, AutoTokenizer, GPTJConfig
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from hf_model_config import PretrainedHFPjitModelConfig
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P

# PartitionSpec for GPTJ
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_gptj():
    return [
        # embeddings
        (("transformer", "wte", "embedding"), P("mp", None)),
        # atention
        (("attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("attn", "out_proj", "kernel"), P("mp", None)),
        # mlp
        (("mlp", "fc_in", "kernel"), P(None, "mp")),
        (("mlp", "fc_in", "bias"), P("mp")),
        (("mlp", "fc_out", "kernel"), P("mp", None)),
        (("mlp", "fc_out", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
        (("lm_head", "bias"), P("mp")), 
    ]

@dataclass
class GPTJModelConfigScript(PretrainedHFPjitModelConfig):
    def unroll(self, metaconfig: MetaConfig):
        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        tokenizer.pad_token = tokenizer.eos_token
        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()
            model, params = FlaxGPTJForCausalLM.from_pretrained(self.model_str, _do_init=False, pad_token_id=tokenizer.eos_token_id, dtype=dtype)
            params = freeze(params)
            params = self.params_to_dtype(model, params)
        rules = _get_partition_rules_gptj()
        return model, params, tokenizer, rules
