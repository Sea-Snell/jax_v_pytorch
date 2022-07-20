from micro_config import MetaConfig, parse_args, deep_replace
from lm_inference_config import LMInferenceConfigScript
from gpt2_config import GPT2ModelConfigScript
from gptj_config import GPTJModelConfigScript
from opt_config import OPTModelConfigScript
from t5_config import T5ModelConfigScript

gpt2_model = GPT2ModelConfigScript(
    model_str="gpt2-xlarge", 
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
    model_str="t5-small", 
    # model_str="google/ul2", 
    use_fp16=True, 
)

pretrained_model = t5_model

lm_inference = LMInferenceConfigScript(
    pretrained_model=pretrained_model, 
    max_len=128, 
    seed=2, 
    n_inferences=25, 
    # prompt='[S2S] hi my friend! <extra_id_0>', 
    prompt='hi my friend!', 
    # prompt='translate English to German: The house is wonderful.', 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        verbose=False, 
    )
    inference = deep_replace(lm_inference, **parse_args())
    inference.unroll(metaconfig)
