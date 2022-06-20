from collections import namedtuple
from typing import List
import numpy as np
from torch.utils.data import Dataset
from logs import LogTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    @staticmethod
    def collate(items):
        imgs, labels = zip(*items)
        imgs, labels = np.array(np.stack(imgs, axis=0)), np.array(np.stack(labels, axis=0))
        imgs = np.reshape(imgs, (-1, 28*28)).astype(np.float32)
        return imgs, labels

class MLP(nn.Module):
    def __init__(self, output_shapes: List[int], dropout_rate: float):
        super().__init__()
        items = []
        for i in range(len(output_shapes)-2):
            items.append(nn.Linear(output_shapes[i], output_shapes[i+1]))
            items.append(nn.Dropout(dropout_rate))
            items.append(nn.ReLU())
        items.append(nn.Linear(output_shapes[-2], output_shapes[-1]))
        self.sequence = nn.Sequential(*items)
    
    def forward(self, x):
        return self.sequence(x)
    
    def loss(self, x, y):
        n = y.shape[0]
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((torch.argmax(predictions, dim=1) == y).float().mean(), n)}
        return loss, logs
