from typing import Optional
import jax
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from load_model_utils import set_partitions, _id_fn
import numpy as np
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P

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
def load_gpt2(model_str, **kwargs):
    model, params = FlaxGPT2LMHeadModel.from_pretrained(model_str, _do_init=False, **kwargs)
    emb = jnp.zeros((50264, model.config.hidden_size))
    emb = emb.at[:50257, :].set(params["transformer"]["wte"]["embedding"])
    params["transformer"]["wte"]["embedding"] = emb
    config = GPT2Config.from_pretrained(model_str, vocab_size=50264, **kwargs)
    model = FlaxGPT2LMHeadModel(config, _do_init=False)
    return model, freeze(params)

@dataclass
class LMInferenceGPT2(ConfigScript):
    model_str: str
    max_len: Optional[int]
    seed: int
    n_inferences: int
    prompt: str

    def unroll(self, metaconfig: MetaConfig):
        rng = jax.random.PRNGKey(self.seed)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_str)
        tokenizer.pad_token = tokenizer.eos_token
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = load_gpt2(self.model_str, pad_token_id=tokenizer.eos_token_id, dtype=jnp.bfloat16)
        params = model.to_bf16(params)
        param_spec = set_partitions(unfreeze(params), _get_partition_rules_gpt2())

        p_get_initial_params = pjit(
            _id_fn, 
            in_axis_resources=None,
            out_axis_resources=(param_spec, None), 
        )

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.local_device_count())

        # actually initialize the opt_state
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