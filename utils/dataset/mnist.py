import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times in the dir from this directory
PATH = os.path.join(ROOT, 'data')  # MNIST() needs only the root directory where the folder `mnist` is located


def load_data(path=None, download=False, flatten=True):
    if path is None:
        path = PATH
    transforms_ = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if flatten:
        transforms_.append(transforms.Lambda(lambda x: torch.flatten(x)))
    transform = transforms.Compose(transforms_)
    train_dataset = MNIST(root=path, train=True, download=download, transform=transform)
    test_dataset = MNIST(root=path, train=False, download=download, transform=transform)

    return train_dataset, test_dataset
