from typing import List, Union
import jax.numpy as jnp
import numpy as np
from optax import softmax_cross_entropy_with_integer_labels
import math
from flax.core.frozen_dict import freeze

def chunk_tokens(tokens: Union[List[int], np.array], seq_len: int, pad_token_id: int) -> np.array:
    tokens = np.asarray(tokens)
    padded_len = math.ceil(len(tokens)/seq_len)*seq_len
    chunked = np.concatenate((tokens, np.full((padded_len-tokens.shape[0],), pad_token_id)), axis=0).reshape(-1, seq_len)
    return chunked

class LMDataset:
    def __init__(self, tokens: np.array):
        self.tokens = tokens
    
    def __getitem__(self, index):
        return freeze({'input_ids': jnp.asarray(self.tokens[index], dtype=jnp.int32)})

    def __len__(self):
        return self.tokens.shape[0]

class Seq2SeqDataset:
    def __init__(self, in_tokens: np.array, out_tokens: np.array):
        assert in_tokens.shape[0] == out_tokens.shape[0]
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
    
    def __getitem__(self, index):
        in_tokens = self.in_tokens[index]
        out_tokens = self.out_tokens[index]
        return freeze({'input_ids': jnp.asarray(in_tokens, dtype=jnp.int32), 'decoder_input_ids': jnp.asarray(out_tokens, dtype=jnp.int32)})
    
    def __len__(self):
        return self.in_tokens.shape[0]

def model_loss(logits, labels, attn_mask):
    loss = (softmax_cross_entropy_with_integer_labels(logits, labels) * attn_mask).sum() / attn_mask.sum()
    logs = {'loss': loss}
    return loss, logs
