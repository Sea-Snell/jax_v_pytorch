from typing import List
import numpy as np
from torch.utils.data import Dataset
from logs import LogTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K

class ImageData(Dataset):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, n_labels: int):
        self.imgs = imgs
        self.labels = labels
        self.n_labels = n_labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

    @staticmethod
    def collate(items):
        imgs, labels = zip(*items)
        imgs, labels = np.array(np.stack(imgs, axis=0)), np.array(np.stack(labels, axis=0))
        return imgs, labels

class DataAugmentation(nn.Module):
    def __init__(self, img_size: int, padding: int):
        super().__init__()
        self.transforms = nn.Sequential(
            K.RandomCrop(size=(img_size, img_size), pad_if_needed=True, padding_mode='replicate', padding=padding), 
            K.RandomHorizontalFlip(p=0.5), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)

class MLP(nn.Module):
    def __init__(self, img_size: int, shapes: List[int], dropout_rate: float, do_aug: bool=True, crop_aug_padding: int=4):
        super().__init__()
        items = []
        for i in range(len(shapes)-2):
            items.append(nn.Linear(shapes[i], shapes[i+1]))
            items.append(nn.Dropout(dropout_rate))
            items.append(nn.ReLU())
        items.append(nn.Linear(shapes[-2], shapes[-1]))
        self.sequence = nn.Sequential(*items)
        self.augmentations = DataAugmentation(img_size=img_size, padding=crop_aug_padding)
        self.do_aug = do_aug
    
    def forward(self, x):
        x = x / 255.0
        if self.training and self.do_aug:
            x = self.augmentations(x)
        x = x.reshape(x.shape[0], -1)
        return self.sequence(x)
    
    def loss(self, x, y):
        n = y.shape[0]
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((torch.argmax(predictions, dim=1) == y).float().mean(), n)}
        return loss, logs

class SimpleCNN(nn.Module):
    def __init__(self, img_size: int, n_channels: int, n_labels: int, do_aug: bool=True, crop_aug_padding: int=4):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(5, 5), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), 
            nn.Dropout(p=0.25), 
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), 
            nn.Dropout(p=0.25), 
            nn.Flatten(start_dim=1, end_dim=-1), 
            nn.Linear(64*(img_size // 4)*(img_size // 4), 128), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(128, n_labels), 
        )
        self.augmentations = DataAugmentation(img_size=img_size, padding=crop_aug_padding)
        self.do_aug = do_aug
    
    def forward(self, x):
        x = x / 255.0
        if self.training and self.do_aug:
            x = self.augmentations(x)
        return self.sequence(x)
    
    def loss(self, x, y):
        n = y.shape[0]
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        logs = {'loss': LogTuple(loss, n), 'acc': LogTuple((torch.argmax(predictions, dim=1) == y).float().mean(), n)}
        return loss, logs
