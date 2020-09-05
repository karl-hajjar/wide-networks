import unittest
import os
import math
import numpy as np

from pytorch.models.resnet_mnist import ResNetMNIST
from pytorch.configs.model import ModelConfig

RESOURCES_DIR = '../resources/'
BATCH_SIZE = 32
BATCH_SIZES = [10, 16, 32, 50, 64, 65, 100]
DATA_DIR = '../../data/'
N_TRAIN = 60000


class TestMNISTLoader(unittest.TestCase):
    def setUp(self) -> None:
        config = ModelConfig(config_file=os.path.join(RESOURCES_DIR, 'resnet_config.yaml'))
        self.resnet = ResNetMNIST(config)

    def test_downloading(self):
        train_data_loader = self.resnet.train_dataloader(download=True)
        print('{:,}'.format(len(train_data_loader.dataset)))

    def test_loading(self):
        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False)
        print(type(train_data_loader.dataset.data))
        print('{:,}'.format(len(train_data_loader.dataset.data)))
        print(type(train_data_loader.dataset.targets))
        print('{:,}'.format(len(train_data_loader.dataset.targets)))
        x, y = train_data_loader.dataset[0]
        print(type(x))
        print(x.shape)
        print(type(y))
        print(y)

    def test_iterating_dataloader(self):
        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE)
        self._check_data_loader_batches(train_data_loader, batch_size=BATCH_SIZE)

    def test_iterating_dataloader_mutiple_bacth_sizes(self):
        for batch_size in BATCH_SIZES:
            print('batch_size :', batch_size)
            train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=batch_size)
            self._check_data_loader_batches(train_data_loader, batch_size=batch_size)
            print('')

    def test_train_val_split(self):
        n_train = int(0.8 * N_TRAIN)
        n_val = N_TRAIN - n_train
        indexes = list(range(N_TRAIN))
        np.random.shuffle(indexes)
        train_indexes = indexes[:n_train]
        val_indexes = indexes[n_train:]

        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                         indexes=train_indexes)
        self.assertTrue(len(train_data_loader.dataset) == n_train)

        val_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                       indexes=val_indexes)
        self.assertTrue(len(val_data_loader.dataset) == n_val)

        self._check_data_loader_batches(train_data_loader, batch_size=BATCH_SIZE)
        self._check_data_loader_batches(val_data_loader, batch_size=BATCH_SIZE)

    def _check_data_loader_batches(self, data_loader, batch_size, verbose=False):
        i = 0
        i_max = 2
        x_shapes = []
        y_shapes = []
        for id, (x, y) in enumerate(data_loader):
            x_shapes.append(x.shape)
            y_shapes.append(y.shape)
            if i < i_max:
                if verbose:
                    print(type(x))
                    print(x.shape)
                    print(type(y))
                    print(y.shape)
                    print(y)
                    print('')
            i += 1

        self.assertTrue(i == len(x_shapes) == len(y_shapes))
        self.assertTrue(len(x_shapes) == math.ceil(len(data_loader.dataset) / batch_size))
        self.assertTrue(all([x_shapes[j][0] == y_shapes[j][0] for j in range(len(x_shapes))]))
        print('last batch shape :', x_shapes[-1])
        self.assertTrue(x_shapes[-1][0] <= batch_size)


if __name__ == '__main__':
    unittest.main()
