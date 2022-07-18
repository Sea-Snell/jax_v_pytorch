from typing import Optional
import jax
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from transformers import GPT2Tokenizer, FlaxOPTForCausalLM, OPTConfig
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from load_model_utils import set_partitions, _id_fn
import numpy as np
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P

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
    model, params = FlaxOPTForCausalLM.from_pretrained(model_str, _do_init=False, dtype=dtype, **kwargs)
    pos_emb = params['model']['decoder']['embed_positions']['embedding'][:2048, :]
    params['model']['decoder']['embed_positions']['embedding'] = pos_emb
    config = OPTConfig.from_pretrained(model_str, dtype=dtype, **kwargs)
    model = FlaxOPTForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

@dataclass
class LMInferenceOPT(ConfigScript):
    model_str: str
    max_len: Optional[int]
    seed: int
    n_inferences: int
    prompt: str

    def unroll(self, metaconfig: MetaConfig):
        rng = jax.random.PRNGKey(self.seed)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_str)
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = load_opt(self.model_str, dtype=jnp.bfloat16)
        params = model.to_bf16(params)
        param_spec = set_partitions(unfreeze(params), _get_partition_rules_opt())

        p_get_initial_params = pjit(
            _id_fn, 
            in_axis_resources=(param_spec, None), 
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
                print(tokenizer.decode(generation))
                print('='*25)
                print()