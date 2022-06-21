from base_configs import MNISTDataConfig, FashionMNISTDataConfig, CIFAR10DataConfig, CIFAR100DataConfig, MLPConfig, SimpleCNNConfig, AdamWConfig, project_root
from torch_configs import DeviceConfig
from train_loop import TrainLoop, StandardaEvaluator
from micro_config import MetaConfig, deep_replace, parse_args
from copy import deepcopy

device = DeviceConfig.gpu_if_available()

# MNIST / FashionMNIST models and data

mnist_train_data = MNISTDataConfig(split='train')
mnist_eval_data = MNISTDataConfig(split='test')

fashion_mnist_train_data = FashionMNISTDataConfig(split='train')
fashion_mnist_eval_data = FashionMNISTDataConfig(split='test')

mnist_mlp_model = MLPConfig(
    shapes=[28*28, 128, 128, 10], 
    dropout=0.5, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

mnist_cnn_model = SimpleCNNConfig(
    img_size=28, 
    n_channels=1, 
    n_labels=10, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

# CIFAR 10 models and data

cifar10_train_data = CIFAR10DataConfig(split='train')
cifar10_eval_data = CIFAR10DataConfig(split='test')

cifar10_mlp_model = MLPConfig(
    shapes=[32*32*3, 128, 128, 10], 
    dropout=0.5, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

cifar10_cnn_model = SimpleCNNConfig(
    img_size=32, 
    n_channels=3, 
    n_labels=10, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

# CIFAR 100 models and data

cifar100_train_data = CIFAR100DataConfig(split='train')
cifar100_eval_data = CIFAR100DataConfig(split='test')

cifar100_mlp_model = MLPConfig(
    shapes=[32*32*3, 128, 128, 100], 
    dropout=0.5, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

cifar100_cnn_model = SimpleCNNConfig(
    img_size=32, 
    n_channels=3, 
    n_labels=100, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

# opimizer, evaluator, training loop

train_data, eval_data, model = None, None, None

optim = AdamWConfig(
    lr=3e-4, 
    weight_decay=0.00, 
    model=model, 
    state_path=None, 
)

evaluator = StandardaEvaluator(
    eval_data=eval_data, 
    model=model, 
    bsize=32, 
    eval_batches=1, 
    dataloader_workers=0, 
    loss_kwargs={}, 
)

train = TrainLoop(
    model=model, 
    train_data=train_data, 
    optim=optim, 
    evaluator=evaluator, 
    save_dir=None, 
    max_checkpoints=1, 
    epochs=10, 
    max_steps=None, 
    bsize=32, 
    grad_accum_steps=1, 
    log_every=4096, 
    eval_every=4096, 
    save_every=None, 
    dataloader_workers=0, 
    use_wandb=False, 
    wandb_project='torch_mnist_test', 
    loss_kwargs={}, 
)

if __name__ == "__main__":
    # Train for 10 epochs on every setting of model and dataset. Comment out the ones you don't want to train.
    runs = [
        ('MLP_MNIST', mnist_train_data, mnist_eval_data, mnist_mlp_model), 
        ('CNN_MNIST', mnist_train_data, mnist_eval_data, mnist_cnn_model), 
        ('MLP_FashionMNIST', fashion_mnist_train_data, fashion_mnist_eval_data, mnist_mlp_model), 
        ('CNN_FashionMNIST', fashion_mnist_train_data, fashion_mnist_eval_data, mnist_cnn_model), 
        ('MLP_CIFAR10', cifar10_train_data, cifar10_eval_data, cifar10_mlp_model), 
        ('CNN_CIFAR10', cifar10_train_data, cifar10_eval_data, cifar10_cnn_model), 
        ('MLP_CIFAR100', cifar100_train_data, cifar100_eval_data, cifar100_mlp_model), 
        ('CNN_CIFAR100', cifar100_train_data, cifar100_eval_data, cifar100_cnn_model), 
    ]

    for name, train_data, eval_data, model in runs:
        print('='*20)
        print('Running:', name)
        print('='*20)
        metaconfig = MetaConfig(
            project_root=project_root, 
            verbose=False, 
        )

        curr_train = deepcopy(train)
        curr_train = deep_replace(curr_train, model=model, 
                                  train_data=train_data, optim={'model': model}, 
                                  evaluator={'model': model, 'eval_data': eval_data})
        curr_train = deep_replace(curr_train, **parse_args())
        curr_train.unroll(metaconfig)
