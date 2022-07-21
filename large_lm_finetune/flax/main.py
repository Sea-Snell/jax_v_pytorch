from micro_config import MetaConfig, deep_replace, parse_args
from model_configs.gptj_config import GPTJModelConfigScript
from model_configs.gpt2_config import GPT2ModelConfigScript
from model_configs.opt_config import OPTModelConfigScript
from model_configs.t5_config import T5ModelConfigScript
from base_configs import AdaFactorConfig, WikitextLMConfig, AdamWConfig, WikitextSeq2SeqConfig, project_root
from flax_hf_configs import RNGSeed
from train_loop import TrainLoop, StandardEvaluator
import jax.numpy as jnp

seed = RNGSeed(0)

seq_len = 1024

gpt2_model = GPT2ModelConfigScript(
    model_str="gpt2", 
    use_fp16=True, 
    params=None, 
)

gptj_model = GPTJModelConfigScript(
    model_str="EleutherAI/gpt-j-6B", 
    use_fp16=True, 
    params=None, 
)

opt_model = OPTModelConfigScript(
    model_str="facebook/opt-350m", 
    use_fp16=True, 
    params=None, 
)

t5_model = T5ModelConfigScript(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-11b", 
    model_str="google/ul2", 
    use_fp16=True, 
    gradient_checkpoint=True, 
    params=None, 
)

model = t5_model

wikitext_lm_train = WikitextLMConfig(
    version="wikitext-2-raw-v1", 
    split="train", 
    max_len=seq_len, 
    model_tokenizer=model, 
)

wikitext_seq2seq_train = WikitextSeq2SeqConfig(
    version="wikitext-2-raw-v1", 
    split="train", 
    enc_len=seq_len//2, 
    dec_len=seq_len//2, 
    model_tokenizer=model, 
)

train_dataset = wikitext_seq2seq_train

wikitext_lm_eval = WikitextLMConfig(
    version="wikitext-2-raw-v1", 
    split="validation", 
    max_len=seq_len, 
    model_tokenizer=model, 
)

wikitext_seq2seq_eval = WikitextSeq2SeqConfig(
    version="wikitext-2-raw-v1", 
    split="validation", 
    enc_len=seq_len//2, 
    dec_len=seq_len//2, 
    model_tokenizer=model, 
)

eval_dataset = wikitext_seq2seq_eval

adamw_optim = AdamWConfig(
    grad_accum_steps=1, 
    lr=1e-5, 
    weight_decay=0.00, 
)

adafactor_optim = AdaFactorConfig(
    grad_accum_steps=1, 
    lr=0.001, 
    multiply_by_parameter_scale=False, 
    dtype_momentum=jnp.bfloat16, 
)

optim = adamw_optim

evaluator = StandardEvaluator(
    eval_data=eval_dataset, 
    model=model, 
    rng=seed.split(1), 
    bsize=1, 
    prefetch_batches=None, 
    eval_batches=16, 
    max_len=seq_len, 
    pjit=True, 
    loss_kwargs={}, 
    verbose=False, 
)

train = TrainLoop(
    train_data=train_dataset, 
    model=model, 
    optim=optim, 
    evaluator=evaluator, 
    rng=seed.split(2), 
    save_dir=None, 
    max_checkpoints=None, 
    epochs=10, 
    max_steps=None, 
    bsize=16, 
    max_len=seq_len, 
    prefetch_batches=None, 
    log_every=16, 
    eval_every=4096, 
    save_every=None, 
    pjit=True, 
    use_wandb=True, 
    wandb_project='pjit_flax_wikitext_finetune', 
    loss_kwargs={}, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    train = deep_replace(train, **parse_args())
    train.unroll(metaconfig)
