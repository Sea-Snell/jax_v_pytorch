from typing import Any
import torch
import tree

def to(item: Any, device: torch.device):
    return tree.map_structure(lambda x: torch.tensor(x).to(device), item)
