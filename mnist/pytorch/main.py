from base_configs import MNISTDataConfig, MLPConfig, MNISTCNNConfig, AdamWConfig, project_root
from torch_configs import DeviceConfig
from train_loop import TrainLoop, StandardaEvaluator
from micro_config import MetaConfig, deep_replace, parse_args

device = DeviceConfig.gpu_if_available()

train_data = MNISTDataConfig(split='train')
eval_data = MNISTDataConfig(split='test')

mlp_model = MLPConfig(
    shapes=[28*28, 128, 128, 10], 
    dropout=0.5, 
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

cnn_model = MNISTCNNConfig(
    checkpoint_path=None, 
    strict_load=True, 
    device=device, 
)

model = cnn_model

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
    save_dir='outputs/mnist_test/', 
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
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    train = deep_replace(train, **parse_args())
    train.unroll(metaconfig)
