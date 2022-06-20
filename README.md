# jax_v_pytorch

A comparison of two different jax frameworks ([haiku](https://dm-haiku.readthedocs.io/en/latest/) and [flax](https://flax.readthedocs.io/en/latest/overview.html)) against [pytorch](https://pytorch.org). Currently MNIST training is the only comparison implemented, but more comparisons will be added later.

## installation

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate jax_v_torch
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate jax_v_torch
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch cudatoolkit=11.3 -c pytorch
```

**install with docker (gpu only):**
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

## Implementations

All implementations are meant to be identical modulo framework specific differences.

* `mnist/pytorch/` implements MNIST training on a simple MLP or CNN in pytorch
* `mnist/flax/` implements MNIST training on a simple MLP or CNN in flax
* `mnist/haiku/` implements MNIST training on a simple MLP or CNN in haiku
