from math import trunc
from typing import Optional
import jax
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze
from load_model_utils import set_partitions, _id_fn
import numpy as np
from jax.experimental.maps import Mesh
from hf_model_config import PretrainedHFPjitModelConfig

@dataclass
class LMInferenceConfigScript(ConfigScript):
    pretrained_model: PretrainedHFPjitModelConfig
    max_prompt_len: Optional[int]
    max_len: Optional[int]
    seed: int
    n_inferences: int
    prompt: str

    def unroll(self, metaconfig: MetaConfig):
        rng = jax.random.PRNGKey(self.seed)
        model, params, tokenizer, rules = self.pretrained_model.unroll(metaconfig)
        # specifies how to split model parameters beteen devices
        param_spec = set_partitions(unfreeze(params), rules)

        # initialization function for splitting parameters to devices
        p_get_initial_params = pjit(
            _id_fn, 
            in_axis_resources=(param_spec, None), 
            out_axis_resources=(param_spec, None), 
        )

        # mesh definition
        mesh_devices = np.array(jax.devices()).reshape(1, jax.device_count())
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)

        # split the params between all devices
        with Mesh(mesh_devices, ("dp", "mp")):
            params, _ = p_get_initial_params(freeze(params), jnp.ones((), dtype=jnp.uint32))
        
        def generate_fn(tokens, attn_mask, params, rng, max_len):
            return model.generate(tokens, attention_mask=attn_mask, max_length=max_len, do_sample=True, prng_key=rng, params=params).sequences[0]

        # model parallel inference function
        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(None, None, param_spec, None), 
            out_axis_resources=None, 
            static_argnums=(4,), 
        )

        # get input tokens
        if self.max_prompt_len is None:
            token_out = tokenizer(self.prompt)
        else:
            token_out = tokenizer(self.prompt, max_length=self.max_prompt_len, padding='max_length', truncation=True)
        tokens = jnp.array(token_out['input_ids'], dtype=jnp.int32)
        attn_mask = jnp.array(token_out['attention_mask'], dtype=jnp.int32)

        # generate sequences
        with Mesh(mesh_devices, ("dp", "mp")):
            for _ in range(self.n_inferences):
                rng, new_rng = jax.random.split(rng)
                generation = p_generate_fn(tokens[None], attn_mask[None], params, new_rng, self.max_len)
                print('='*25)
                print('input:')
                print(tokenizer.decode(tokens))
                print('='*25)
                print('model generation:')
                print(tokenizer.decode(generation))
                print('='*25)
                print()
