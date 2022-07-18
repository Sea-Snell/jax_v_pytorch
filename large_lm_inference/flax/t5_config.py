import jax
import jax.numpy as jnp
from transformers import FlaxT5ForConditionalGeneration, T5Config, AutoTokenizer, T5ForConditionalGeneration
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from jax.experimental import PartitionSpec as P
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from hf_model_config import PretrainedHFPjitModelConfig

# PartitionSpec for T5v1.1
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5_v1_1():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for UL2
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_ul2():
    return [
        # embeddings
        (('encoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (('decoder', 'embed_tokens', 'kernel'), P("mp", None)), 
        (("shared", "embedding"), P("mp", None)), 
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi_0", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wi_1", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

# PartitionSpec for T5
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_t5():
    return [
        # embeddings
        (("shared", "embedding"), P("mp", None)),
        (("relative_attention_bias", "embedding"), None), 
        # self atention
        (("SelfAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("SelfAttention", "o", "kernel"), P("mp", None)),
        # cross atention
        (("EncDecAttention", "(k|q|v)", "kernel"), P(None, "mp")),
        (("EncDecAttention", "o", "kernel"), P("mp", None)),
        # mlp
        (("DenseReluDense", "wi", "kernel"), P(None, "mp")),
        (("DenseReluDense", "wo", "kernel"), P("mp", None)),
        # layer norms
        (("layer_norm", "weight"), None), 
        (("final_layer_norm", "weight"), None), 
        # output head
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

def load_t5(model_str, dtype=jnp.float32, **kwargs):
    if model_str == 'google/ul2':
        # have to load through pytorch and convert weights manually due to bug with transformers for partitioned weights
        # see: https://github.com/huggingface/transformers/pull/18170
        pytorch_model = T5ForConditionalGeneration.from_pretrained(model_str, **kwargs)
        config = T5Config.from_pretrained(model_str, dtype=dtype, **kwargs)
        model = FlaxT5ForConditionalGeneration(config, dtype=dtype, **kwargs)
        params = convert_pytorch_state_dict_to_flax(pytorch_model.state_dict(), model)
    else:
        try:
            model, params = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=False, dtype=dtype, **kwargs)
        except:
            model = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=True, from_pt=True, dtype=dtype, **kwargs)
            params = model.params
            config = T5Config.from_pretrained(model_str, dtype=dtype, **kwargs)
            model = FlaxT5ForConditionalGeneration(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

@dataclass
class T5ModelConfigScript(PretrainedHFPjitModelConfig):
    def unroll(self, metaconfig: MetaConfig):
        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()
            model, params = load_t5(self.model_str, dtype=dtype)
            params = self.params_to_dtype(model, params)
        if 'v1_1' in self.model_str:
            rules = _get_partition_rules_t5_v1_1()
        elif self.model_str == 'google/ul2':
            rules = _get_partition_rules_ul2()
        else:
            rules = _get_partition_rules_t5()
        return model, params, tokenizer, rules
