from typing import Any, Dict, List, Optional
import tree
import numpy as np
import torch
import wandb
from functools import reduce
from dataclasses import dataclass

PyTree = Any

@dataclass
class LogTuple:
    mean: float
    count: int

def is_scalar(x):
    return isinstance(x, int) or isinstance(x, float) or (isinstance(x, torch.Tensor) and len(x.shape) == 0) or (isinstance(x, np.ndarray) and len(x.shape) == 0)

def is_vector(x):
    return (isinstance(x, np.ndarray) and len(x.shape) > 0) or (isinstance(x, torch.Tensor) and len(x.shape) > 0)

def is_leaf(x):
    return is_vector(x) or is_scalar(x) or isinstance(x, LogTuple)

def un_torch_logs(logs):
    def un_torch_log_f(x):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.tolist()
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 0:
                return x.detach().cpu().item()
            else:
                return x.detach().cpu().tolist()
        return x
    return tree.map_structure(un_torch_log_f, logs)

def reduce_elements(x):
    if isinstance(x, LogTuple):
        return x.mean
    if is_vector(x):
        return x.mean()
    if is_scalar(x):
        return x
    raise NotImplementedError

def combine_elements(a, b):
    if is_scalar(a):
        a = LogTuple(a, 1)
    if is_scalar(b):
        b = LogTuple(b, 1)
    if isinstance(a, LogTuple) and isinstance(b, LogTuple):
        if (a.count + b.count) == 0:
            return LogTuple(0.0, 0)
        return LogTuple((a.mean * a.count + b.mean * b.count) / (a.count + b.count), a.count + b.count)
    if is_vector(a) and is_vector(b):
        return torch.cat((a, b,), axis=0)
    raise NotImplementedError

def reduce_logs(logs: List[PyTree], initial_log: Optional[PyTree]=None) -> PyTree:
    if initial_log is None:
        return tree.map_structure(lambda *x: reduce(combine_elements, x), *logs)
    return tree.map_structure(lambda *x: reduce(combine_elements, x, initial_log), *logs)

def pool_logs(logs: PyTree) -> Any:
    logs = tree.map_structure(reduce_elements, logs)
    logs = un_torch_logs(logs)
    return logs

def label_logs(logs: Any, label: str, to_add: Dict[str, Any]) -> Dict[str, Any]:
    return {label: logs, **to_add}

def log(logs: Any, use_wandb: bool) -> None:
    if use_wandb:
        wandb.log(logs)
    print(logs)
