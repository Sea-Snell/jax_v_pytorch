# jax_v_pytorch

A comparison of two different jax frameworks ([haiku](https://dm-haiku.readthedocs.io/en/latest/) and [flax](https://flax.readthedocs.io/en/latest/overview.html)) against [pytorch](https://pytorch.org). Currently implements MNIST, FashionMNIST, CIFAR10, and CIFAR100 training on MLPs and CNNs.

## installation

**install with conda (cpu):**
``` shell
git clone https://github.com/Sea-Snell/jax_v_pytorch.git
cd jax_v_pytorch
conda env create -f environment.yml
conda activate jax_v_torch
```

**install with conda (gpu):**
``` shell
git clone https://github.com/Sea-Snell/jax_v_pytorch.git
cd jax_v_pytorch
conda env create -f environment.yml
conda activate jax_v_torch
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with docker (gpu only):**
* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
git clone https://github.com/Sea-Snell/jax_v_pytorch.git
cd jax_v_pytorch
docker compose build
docker compose run jax_v_torch
```

## Running

1. navigate to any subfolder (for example `cd cifar_mnist/haiku/`)
2. `python main.py`

Feel free to edit any configs in `main.py` by editing the file or with command line arguments. The config framework is [micro-config](https://github.com/Sea-Snell/micro_config).

## Implementations

All implementations are meant to be identical modulo framework specific differences.

* `cifar_mnist/pytorch/` implements MNIST/FashionMNIST/CIFAR10/CIFAR100 training on a simple MLP or CNN in pytorch
* `cifar_mnist/flax/` implements MNIST/FashionMNIST/CIFAR10/CIFAR100 training on a simple MLP or CNN in flax
* `cifar_mnist/haiku/` implements MNIST/FashionMNIST/CIFAR10/CIFAR100 training on a simple MLP or CNN in haiku
