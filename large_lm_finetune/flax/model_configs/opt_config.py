import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer, FlaxOPTForCausalLM, OPTConfig, OPTForCausalLM
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from .hf_model import PretrainedHFPjitModelConfig
from jax.experimental import PartitionSpec as P
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

# PartitionSpec for OPT
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_opt():
    return [
        # embeddings
        (("model", "decoder", "embed_positions", "embedding"), P("mp", None)),
        (("model", "decoder", "embed_tokens", "embedding"), P("mp", None)),
        (("model", "decoder", "project_in", "kernel"), None), 
        (("model", "decoder", "project_out", "kernel"), None), 
        # atention
        (("self_attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("self_attn", "(k_proj|q_proj|v_proj)", "bias"), P("mp")),
        (("self_attn", "out_proj", "kernel"), P("mp", None)),
        (("self_attn", "out_proj", "bias"), P(None)),
        # mlp
        (("fc1", "kernel"), P(None, "mp")),
        (("fc1", "bias"), P("mp")),
        (("fc2", "kernel"), P("mp", None)),
        (("fc2", "bias"), None),
        # layer norms
        (("final_layer_norm", "bias"), None),
        (("final_layer_norm", "scale"), None),
        (("self_attn_layer_norm", "bias"), None),
        (("self_attn_layer_norm", "scale"), None),
        # output head
        (("model", "lm_head", "kernel"), P(None, "mp")), 
    ]

def load_opt(model_str, dtype=jnp.float32, **kwargs):
    partitioned_models = ['facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']
    if model_str in partitioned_models:
        # have to load through pytorch and convert weights manually due to bug with transformers for partitioned weights
        # see: https://github.com/huggingface/transformers/pull/18170
        pytorch_model = OPTForCausalLM.from_pretrained(model_str, **kwargs)
        config = OPTConfig.from_pretrained(model_str, dtype=dtype, **kwargs)
        model = FlaxOPTForCausalLM(config, dtype=dtype, **kwargs)
        params = convert_pytorch_state_dict_to_flax(pytorch_model.state_dict(), model)
    else:
        model, params = FlaxOPTForCausalLM.from_pretrained(model_str, _do_init=False, dtype=dtype, **kwargs)
    pos_emb = jnp.zeros((4096, model.config.hidden_size))
    pos_emb = pos_emb.at[:2050, :].set(params['model']['decoder']['embed_positions']['embedding'])
    params['model']['decoder']['embed_positions']['embedding'] = pos_emb
    config = OPTConfig.from_pretrained(model_str, max_position_embeddings=4094, dtype=dtype, **kwargs)
    model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

@dataclass
class OPTModelConfigScript(PretrainedHFPjitModelConfig):
    def unroll(self, metaconfig: MetaConfig):
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_str)
        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()
            model, params = load_opt(self.model_str, dtype=dtype)
            params = self.params_to_dtype(model, params)
        rules = _get_partition_rules_opt()
        return model, params, tokenizer, rules
