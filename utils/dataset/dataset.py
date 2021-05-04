import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np

from .random import RandomData


class RandomDataset(Dataset):
    """
    A class defining a Pytorch Dataset. In this instance, the class is implemented as a wrapper around the RandomData
    class. As such the `data` argument to the __init__ method is expected to be an object of the class RandomData.
    """
    def __init__(self, data: RandomData, transform=torch.Tensor):
        super(RandomDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index) -> [(np.array, int), (np.array, np.array), (torch.Tensor, torch.Tensor)]:
        x = self.data.data_points[index, :]
        y = self.data.data_points_labels[index]
        if self.transform is not None:
            # don't change the type here as the bce loss requires float type anyways
            x, y = self.transform(x), self.transform([y])
        return x, y

    def __len__(self):
        return len(self.data.data_points_labels)

    def __add__(self, other):
        return ConcatDataset([self, other])
