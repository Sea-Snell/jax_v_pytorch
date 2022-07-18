from micro_config import MetaConfig, parse_args, deep_replace
from t5_inference import LMInferenceT5
from gpt2_inference import LMInferenceGPT2

gpt2_inference = LMInferenceGPT2(
    model_str="gpt2-large", 
    max_len=128, 
    seed=2, 
    n_inferences=25, 
    prompt='hi my friend!', 
)

t5_inference = LMInferenceT5(
    # model_str="google/t5-v1_1-xl", 
    model_str="t5-3b", 
    max_len=128, 
    seed=2, 
    n_inferences=25, 
    prompt='translate English to German: The house is wonderful.', 
    # prompt='hi my friend!', 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        verbose=False, 
    )
    inference = deep_replace(t5_inference, **parse_args())
    inference.unroll(metaconfig)
