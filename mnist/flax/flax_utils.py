import jax

def rngs_from_keys(rng, keys):
    rngs = {}
    for k in keys:
        rng, new_rng = jax.random.split(rng)
        rngs[k] = new_rng
    return rngs
