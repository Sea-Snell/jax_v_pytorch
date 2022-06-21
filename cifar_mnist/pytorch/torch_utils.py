from typing import Any, List, Set, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import tree

def to(item: Any, device: torch.device):
    return tree.map_structure(lambda x: torch.tensor(x).to(device), item)
