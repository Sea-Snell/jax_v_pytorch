import jax
import jax.numpy as jnp
from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from micro_config import MetaConfig
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from jax.experimental import PartitionSpec as P
from .hf_model import PretrainedHFPjitModelConfig

# PartitionSpec for GPT2
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_gpt2():
    return [
        # embeddings
        (("transformer", "wpe", "embedding"), P("mp", None)),
        (("transformer", "wte", "embedding"), P("mp", None)),
        # atention
        (("attn", "(q_attn|c_attn)", "kernel"), P(None, "mp")),
        (("attn", "(q_attn|c_attn)", "bias"), P("mp")),
        (("attn", "c_proj", "kernel"), P("mp", None)),
        (("attn", "c_proj", "bias"), None),
        # mlp
        (("mlp", "c_fc", "kernel"), P(None, "mp")),
        (("mlp", "c_fc", "bias"), P("mp")),
        (("mlp", "c_proj", "kernel"), P("mp", None)),
        (("mlp", "c_proj", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
    ]

# Source: https://github.com/huggingface/transformers/tree/main/examples/research_projects/jax-projects/model_parallel
def load_gpt2(model_str, dtype=jnp.float32, **kwargs):
    model, params = FlaxGPT2LMHeadModel.from_pretrained(model_str, _do_init=False, dtype=dtype, **kwargs)
    emb = jnp.zeros((50264, model.config.hidden_size))
    emb = emb.at[:50257, :].set(params["transformer"]["wte"]["embedding"])
    params["transformer"]["wte"]["embedding"] = emb
    config = GPT2Config.from_pretrained(model_str, vocab_size=50264, dtype=dtype, **kwargs)
    model = FlaxGPT2LMHeadModel(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

class GPT2ModelConfigScript(PretrainedHFPjitModelConfig):
    def unroll(self, metaconfig: MetaConfig):
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_str)
        tokenizer.pad_token = tokenizer.eos_token
        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()
            model, params = load_gpt2(self.model_str, dtype=dtype, pad_token_id=tokenizer.eos_token_id)
            params = self.params_to_dtype(model, params)
        rules = _get_partition_rules_gpt2()
        return model, params, tokenizer, rules
