# jax_v_pytorch

A comparison of two different jax frameworks ([haiku](https://dm-haiku.readthedocs.io/en/latest/) and [flax](https://flax.readthedocs.io/en/latest/overview.html)) against [pytorch](https://pytorch.org). Currently MNIST training is the only comparison implemented, but more comparisons will be implemented.

## installation

install with conda (cpu only):

``` shell
conda env create -f environment.yml
conda activate jax_v_torch
```

or install with docker (gpu only):

* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run jax_v_torch
```
## Running

1. navigate to any subfolder (for example `cd mnist/haiku/`)
2. `python main.py`

Feel free to edit any configs in `main.py` by editing the file or with command line arguments. The config framework is [micro-config](https://github.com/Sea-Snell/micro_config).
