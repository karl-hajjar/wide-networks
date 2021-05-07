import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times in the dir from this directory
PATH = os.path.join(ROOT, 'data')  # CIFAR10() needs only the root directory where the folder `cifar10` is located


def load_data(path=None, download=False, grayscale=True, flatten=True):
    if path is None:
        path = PATH

    transforms_ = [transforms.ToTensor()]
    if grayscale:
        transforms_.insert(0, transforms.Grayscale(num_output_channels=1))
        transforms_.append(transforms.Normalize(mean=(0.4809,), std=(0.2392,)))
    else:
        transforms_.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if flatten:
        transforms_.append(transforms.Lambda(lambda x: torch.flatten(x)))

    transform = transforms.Compose(transforms_)
    train_dataset = CIFAR10(root=path, train=True, download=download, transform=transform)
    test_dataset = CIFAR10(root=path, train=False, download=download, transform=transform)

    return train_dataset, test_dataset
