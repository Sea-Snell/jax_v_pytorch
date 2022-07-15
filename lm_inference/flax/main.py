from typing import Set
import jax
import jax.numpy as jnp
from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer
from flax.core.frozen_dict import freeze, unfreeze
from flax_utils import rngs_from_keys
from jaxtyping import i32, PyTree
from functools import partial

def init_cache(module, batch_size, max_length):
    r"""
    Args:
        batch_size (`int`):
            batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
        max_length (`int`):
            maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
            cache.
    Source: https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/gpt2/modeling_flax_gpt2.py#L439
    """
    # init input variables to retrieve cache
    input_ids = jnp.ones((batch_size, max_length))
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

    init_variables = module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
    return init_variables["cache"]

def get_sample_fn(module):
    @partial(jax.jit, static_argnums=(6,))
    def sample_from_gpt2(params: PyTree, cache: PyTree, tokens: i32['max_t'], 
                         attn_mask: i32['max_t'], pos_ids: i32['t'], rng: jax.random.PRNGKey, rng_keys: Set[str]):
        t, max_len = pos_ids.shape[0], attn_mask.shape[0]
        rng, new_rng = jax.random.split(rng)
        rngs = rngs_from_keys(new_rng, rng_keys)
        _, state = module.apply(freeze({'params': params, 'cache': cache}), tokens[None, :(t-1)], attn_mask[None], 
                                pos_ids[None, :-1], rngs=rngs, mutable=['cache'], deterministic=True)
        cache = state['cache']

        def sample_fn(i, state):
            tokens, cache, rng = state
            outputs, state = module.apply(freeze({'params': params, 'cache': cache}), tokens[None, i], attn_mask[None], 
                                          i[None, None], rngs=rngs, mutable=['cache'], deterministic=True)
            logits = outputs.logits
            rng, new_rng = jax.random.split(rng)
            tokens = tokens.at[i+1].set(jax.random.categorical(new_rng, logits[0, 0]))
            cache = state['cache']
            return tokens, cache, rng
        
        rng, new_rng = jax.random.split(rng)
        out_tokens, _, _ = jax.lax.fori_loop(t, max_len, sample_fn, (tokens, cache, new_rng,))
        return out_tokens
    return sample_from_gpt2
    

if __name__ == "__main__":
    model = "gpt2"
    max_len = 128
    rng = jax.random.PRNGKey(2)
    prompt = 'hi my friend!'

    tokenizer = GPT2Tokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    model = FlaxGPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    tokens = jnp.array(tokenizer(prompt)['input_ids'], dtype=jnp.int32)
    generation = model.generate(tokens[None], max_length=max_len, do_sample=True, prng_key=rng).sequences
    print(tokenizer.decode(generation[0]))
