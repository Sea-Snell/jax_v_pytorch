from micro_config import MetaConfig, deep_replace, parse_args
from model_configs.gptj_config import GPTJModelConfigScript
from model_configs.gpt2_config import GPT2ModelConfigScript
from model_configs.opt_config import OPTModelConfigScript
from model_configs.t5_config import T5ModelConfigScript
from base_configs import WikitextLMConfig, AdamWConfig, WikitextSeq2SeqConfig, project_root
from flax_hf_configs import RNGSeed
from train_loop import TrainLoop

seed = RNGSeed(0)

seq_len = 1024

gpt2_model = GPT2ModelConfigScript(
    model_str="gpt2", 
    use_fp16=True, 
)

gptj_model = GPTJModelConfigScript(
    model_str="EleutherAI/gpt-j-6B", 
    use_fp16=True, 
)

opt_model = OPTModelConfigScript(
    model_str="facebook/opt-350m", 
    use_fp16=True, 
)

t5_model = T5ModelConfigScript(
    # model_str="google/t5-v1_1-xl", 
    model_str="t5-11b", 
    # model_str="google/ul2", 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

model = t5_model

wikitext_lm = WikitextLMConfig(
    version="wikitext-2-raw-v1", 
    split="train", 
    max_len=seq_len, 
    model_tokenizer=model, 
)

wikitext_seq2seq = WikitextSeq2SeqConfig(
    version="wikitext-2-raw-v1", 
    split="train", 
    enc_len=seq_len//2, 
    dec_len=seq_len//2, 
    model_tokenizer=model, 
)

dataset = wikitext_seq2seq

optim = AdamWConfig(
    grad_accum_steps=8, 
    lr=1e-5, 
    weight_decay=0.00, 
)

train = TrainLoop(
    train_data=dataset, 
    model=model, 
    optim=optim, 
    rng=seed, 
    save_dir=None, 
    max_checkpoints=None, 
    epochs=10, 
    max_steps=None, 
    bsize=1, 
    max_len=seq_len, 
    prefetch_batches=None, 
    log_every=16, 
    save_every=None, 
    pjit=True, 
    use_wandb=False, 
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
