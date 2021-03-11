import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import os

from .base_model import BaseModel
from utils.nn import smoothed_exp_margin, smoothed_logistic_margin
from pytorch.initializers import INIT_DICT

# TODO : issue with torchvision is : architecture is fixed and the layer widths cannot be modified
#  see here for a useful re-implmentation : https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
#  Idea : re-implement the resenet18 but with modular widths so that multiple experiments can be run with different
#  widths


class ResNet18(BaseModel):
    """A model implementing the resnet18 architecture using the torchivision.models.resnet18 pre-built architecture"""

    def __init__(self, config):
        super().__init__(config)
        self.model = resnet18(pretrained=False, progress=True)

    def train_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = CIFAR100(data_dir, train=True, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def val_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = CIFAR100(data_dir, train=True, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def test_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = CIFAR100(data_dir, train=False, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)