# jax_v_pytorch

Side-by-side implementations of two different jax frameworks ([haiku](https://dm-haiku.readthedocs.io/en/latest/) and [flax](https://flax.readthedocs.io/en/latest/overview.html)) and [pytorch](https://pytorch.org) on simple deep learning training and inference tasks. Currently implements MNIST, FashionMNIST, CIFAR10, and CIFAR100 training on MLPs and CNNs.

## installation

### **1. pull from github**

``` python
git clone https://github.com/Sea-Snell/jax_v_pytorch.git
cd jax_v_pytorch
```

### **2. install dependencies**

Install with conda (cpu or gpu) or docker (gpu only).

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
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate jax_v_torch
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with docker (gpu only):**
* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run jax_v_torch
```

And then in the new container shell that pops up:

``` shell
cd jax_v_torch
```

## Running

1. navigate to any subfolder (for example `cd cifar_mnist/haiku/`)
2. `python main.py`

Feel free to edit any configs in `main.py`. You can do this by either directly editing the file or with command line arguments. The config framework is [micro-config](https://github.com/Sea-Snell/micro_config).

## Implementations

All implementations are meant to be identical modulo framework specific differences.

* `cifar_mnist/` implements MNIST/FashionMNIST/CIFAR10/CIFAR100 training on both single and multiple devices (data parallel).
    * `pytorch/` implemented in pytorch, single device
    * `flax/` implemented in flax, single device
    * `flax_pmap/` implemented in flax, multi device
    * `haiku/` implemented in haiku, single device
