from base_configs import MNISTDataConfig, MLPConfig, AdamWConfig, project_root
from haiku_configs import RNGSeed
from train_loop import TrainLoop, StandardaEvaluator
from micro_config import MetaConfig, deep_replace, parse_args

seed = RNGSeed(0)

train_data = MNISTDataConfig(split='train')
eval_data = MNISTDataConfig(split='test')

model = MLPConfig(
    shapes=[28*28, 128, 128, 10], 
    dropout=0.5, 
    rng=seed.split(1), 
    checkpoint_path=None, 
    params=None, 
    state=None, 
)

optim = AdamWConfig(
    lr=3e-4, 
    weight_decay=0.00, 
    grad_accum_steps=1, 
    model=model, 
    state_path=None, 
    optim_state=None, 
)

evaluator = StandardaEvaluator(
    eval_data=eval_data, 
    model=model, 
    rng=seed.split(2), 
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
    rng=seed.split(3), 
    save_dir='outputs/mnist_test/', 
    max_checkpoints=1, 
    epochs=10, 
    max_steps=None, 
    bsize=32, 
    log_every=4096, 
    eval_every=4096, 
    save_every=None, 
    dataloader_workers=0, 
    jit=True, 
    use_wandb=False, 
    wandb_project='haiku_mnist_test', 
    loss_kwargs={}, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    train = deep_replace(train, **parse_args())
    train.unroll(metaconfig)
