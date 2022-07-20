from typing import Optional
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from datasets import load_dataset
from model_configs.hf_model import PretrainedHFPjitModelConfig
from src import LMDataset, Seq2SeqDataset, chunk_tokens
import optax
import os
import numpy as np

project_root = os.path.dirname(__file__)

@dataclass
class WikitextLMConfig(ConfigScript):
    version: str
    split: str
    max_len: int    
    model_tokenizer: PretrainedHFPjitModelConfig

    def __postinit__(self):
        assert self.version in ["wikitext-103-raw-v1", "wikitext-2-raw-v1"]
        assert self.split in ["train", "valid", "test"]

    def unroll(self, metaconfig: MetaConfig) -> LMDataset:
        dataset = load_dataset("wikitext", self.version, split=self.split)
        all_text = ''.join(map(lambda x: x['text'], dataset))
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        tokens = np.array(tokenizer(all_text)['input_ids'])
        tokens = chunk_tokens(tokens, self.max_len, tokenizer.pad_token_id)
        return LMDataset(tokens)

@dataclass
class WikitextSeq2SeqConfig(ConfigScript):
    version: str
    split: str
    enc_len: int   
    dec_len: int 
    model_tokenizer: PretrainedHFPjitModelConfig

    def __postinit__(self):
        assert self.version in ["wikitext-103-raw-v1", "wikitext-2-raw-v1"]
        assert self.split in ["train", "valid", "test"]

    def unroll(self, metaconfig: MetaConfig) -> LMDataset:
        dataset = load_dataset("wikitext", self.version, split=self.split)
        all_text = ''.join(map(lambda x: x['text'], dataset))
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        tokens = np.array(tokenizer(all_text)['input_ids'])
        tokens = chunk_tokens(tokens, self.enc_len+self.dec_len, tokenizer.pad_token_id)
        in_tokens, out_tokens = tokens[:, :self.enc_len], tokens[:, self.enc_len:]
        return Seq2SeqDataset(in_tokens, out_tokens)

@dataclass
class AdamWConfig(ConfigScript):
    lr: float
    weight_decay: float
    grad_accum_steps: int

    def unroll(self, metaconfig: MetaConfig) -> optax.GradientTransformation:
        optimizer = optax.adamw(self.lr, weight_decay=self.weight_decay)
        optimizer = optax.MultiSteps(optimizer, 
                                     self.grad_accum_steps, 
                                     use_grad_mean=True)
        return optimizer

