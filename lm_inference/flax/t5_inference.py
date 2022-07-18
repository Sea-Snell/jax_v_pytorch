from typing import Optional
import jax
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from load_model_utils import set_partitions, _id_fn
import numpy as np
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P

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

def load_t5(model_str, **kwargs):
    try:
        model, params = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=False, **kwargs)
    except:
        model = FlaxT5ForConditionalGeneration.from_pretrained(model_str, _do_init=True, from_pt=True, **kwargs)
        params = model.params
        config = T5Config.from_pretrained(model_str, **kwargs)
        model = FlaxT5ForConditionalGeneration(config, _do_init=False)
    return model, freeze(params)

@dataclass
class LMInferenceT5(ConfigScript):
    model_str: str
    max_len: Optional[int]
    seed: int
    n_inferences: int
    prompt: str

    def unroll(self, metaconfig: MetaConfig):
        rng = jax.random.PRNGKey(self.seed)
        tokenizer = T5Tokenizer.from_pretrained(self.model_str)
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = load_t5(self.model_str, dtype=jnp.bfloat16)
        params = model.to_bf16(params)
        param_spec = set_partitions(unfreeze(params), 
                                    _get_partition_rules_t5_v1_1() if 'v1_1' in self.model_str else _get_partition_rules_t5())

        p_get_initial_params = pjit(
            _id_fn, 
            in_axis_resources=None,
            out_axis_resources=(param_spec, None), 
        )

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)

        # actually initialize the params
        with Mesh(mesh_devices, ("dp", "mp")):
            params, _ = p_get_initial_params(freeze(params), jnp.ones((), dtype=jnp.uint32))

        def generate_fn(tokens, params, rng, max_len):
            return model.generate(tokens, max_length=max_len, do_sample=True, prng_key=rng, params=params).sequences[0]

        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(None, param_spec, None), 
            out_axis_resources=None, 
            static_argnums=(3,), 
        )

        tokens = jnp.array(tokenizer(self.prompt)['input_ids'], dtype=jnp.int32)

        with Mesh(mesh_devices, ("dp", "mp")):
            for _ in range(self.n_inferences):
                rng, new_rng = jax.random.split(rng)
                generation = p_generate_fn(tokens[None], params, new_rng, self.max_len)
                print('='*25)
                print('input:')
                print(tokenizer.decode(tokens))
                print('='*25)
                print('output:')
                print(tokenizer.decode(generation))
                print('='*25)
                print()