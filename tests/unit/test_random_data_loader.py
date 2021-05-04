import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import math

from utils.dataset import random, dataset


BATCH_SIZE = 32


class TestRandomDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.rs = [0.5, 1, 2]
        self.ks = [2, 3, 4, 5, 10, 15]
        self.ds = [2, 3, 4, 5, 10, 20, 100, 1000]
        self.ns = [10, 50, 100, 500, 1000]

    def test_dataset_without_transform(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = random.RandomData(k=k, r=r, d=d)
                        x, _, y = data.generate_samples(n)
                        ds = dataset.RandomDataset(data, transform=None)
                        self.assertTrue(len(ds) == len(data.data_points_labels) == n)
                        indexes = np.random.choice(np.arange(len(ds)), size=max(10, len(ds) // 10))
                        for index in indexes:
                            x, y = ds[index]
                            self.assertSequenceEqual(x.shape, (d,))
                            self.assertTrue(y in [-1, 1])
                            np.testing.assert_allclose(x, data.data_points[index, :], rtol=1e-9, atol=1e-9)
                            self.assertTrue(y == data.data_points_labels[index])

    def test_dataset_with_transform(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = random.RandomData(k=k, r=r, d=d)
                        X, _, Y = data.generate_samples(n)
                        ds = dataset.RandomDataset(data)
                        self.assertTrue(len(ds) == len(data.data_points_labels) == n)
                        indexes = np.random.choice(np.arange(len(ds)), size=max(10, len(ds) // 10))
                        for index in indexes:
                            x, y = ds[index]
                            self.assertTrue(x.dtype == torch.float32)
                            self.assertTrue(y.dtype == torch.float32)
                            self.assertSequenceEqual(x.shape, (d,))
                            self.assertTrue(y.detach().item() in [-1., 1.])
                            np.testing.assert_allclose(x.detach().cpu().numpy(), data.data_points[index, :],
                                                       rtol=1e-6, atol=1e-6)
                            self.assertTrue(y.detach().item() == data.data_points_labels[index])

    def test_dataloader_without_shuffle(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = random.RandomData(k=k, r=r, d=d)
                        x, _, y = data.generate_samples(n)
                        ds = dataset.RandomDataset(data)
                        dataloader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)
                        i = 0
                        x_shapes = []
                        y_shapes = []
                        for index, (x, y) in enumerate(dataloader):
                            self.assertTrue(index == i)

                            x_shape = x.shape
                            y_shape = y.shape
                            self.assertTrue(len(x_shape) == 2)
                            self.assertTrue(len(y_shape) == 2)
                            self.assertTrue(x_shape[1] == d)
                            self.assertTrue(y_shape[1] == 1)
                            self.assertTrue(x.dtype == torch.float32)
                            self.assertTrue(y.dtype == torch.float32)
                            self.assertTrue((y == 1).sum() + (y == -1).sum() == len(y))
                            np.testing.assert_allclose(x.detach().cpu().numpy(),
                                                       data.data_points[index * BATCH_SIZE: (index+1) * BATCH_SIZE, :],
                                                       rtol=1e-6, atol=1e-6)

                            self.assertTrue((y.detach().cpu().numpy()[:, 0] ==
                                            data.data_points_labels[index * BATCH_SIZE: (index+1) * BATCH_SIZE]).all())
                            x_shapes.append(x_shape)
                            y_shapes.append(y_shape)
                            i += 1

                        self.assertTrue(i == len(x_shapes) == len(y_shapes))
                        self.assertTrue(len(x_shapes) == math.ceil(len(dataloader.dataset) / BATCH_SIZE))
                        self.assertTrue(all([x_shapes[j][0] == y_shapes[j][0] for j in range(len(x_shapes))]))
                        self.assertTrue(x_shapes[-1][0] <= BATCH_SIZE)
                        self.assertTrue(all([y_shapes[j] == (BATCH_SIZE, 1) for j in range(len(x_shapes) - 1)]))

    def test_dataloader_with_shuffle(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = random.RandomData(k=k, r=r, d=d)
                        x, _, y = data.generate_samples(n)
                        ds = dataset.RandomDataset(data)
                        dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
                        i = 0
                        x_shapes = []
                        y_shapes = []
                        for index, (x, y) in enumerate(dataloader):
                            self.assertTrue(index == i)

                            x_shape = x.shape
                            y_shape = y.shape
                            self.assertTrue(len(x_shape) == 2)
                            self.assertTrue(len(y_shape) == 2)
                            self.assertTrue(x_shape[1] == d)
                            self.assertTrue(y_shape[1] == 1)
                            self.assertTrue(x.dtype == torch.float32)
                            self.assertTrue(y.dtype == torch.float32)
                            self.assertTrue((y == 1).sum() + (y == -1).sum() == len(y))
                            self.assertFalse((np.mean(np.abs(x.detach().cpu().numpy() -
                                             data.data_points[index * BATCH_SIZE: (index+1) * BATCH_SIZE, :])) < 1e-1))

                            x_shapes.append(x_shape)
                            y_shapes.append(y_shape)
                            i += 1

                        self.assertTrue(i == len(x_shapes) == len(y_shapes))
                        self.assertTrue(len(x_shapes) == math.ceil(len(dataloader.dataset) / BATCH_SIZE))
                        self.assertTrue(all([x_shapes[j][0] == y_shapes[j][0] for j in range(len(x_shapes))]))
                        self.assertTrue(x_shapes[-1][0] <= BATCH_SIZE)
                        self.assertTrue(all([y_shapes[j] == (BATCH_SIZE, 1) for j in range(len(x_shapes) - 1)]))

    def test_train_val_test_split(self):
        r, k, d, n = 1., 4, 20, 10000
        n_train = 6000
        n_val = 2000
        n_test = 2000

        shuffled_indexes = np.arange(n)
        np.random.shuffle(shuffled_indexes)
        train_indexes = shuffled_indexes[:n_train]
        val_indexes = shuffled_indexes[n_train: n_train + n_val]
        test_indexes = shuffled_indexes[n_train + n_val:]

        data = random.RandomData(k=k, r=r, d=d)
        data.generate_samples(n)
        ds = dataset.RandomDataset(data)

        training_dataset = Subset(ds, train_indexes)
        val_dataset = Subset(ds, val_indexes)
        test_dataset = Subset(ds, test_indexes)

        self.assertTrue(len(training_dataset) == n_train)
        self.assertTrue(len(val_dataset) == n_val)
        self.assertTrue(len(test_dataset) == n_test)

        train_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

        i = 0
        for index, (x,y) in enumerate(train_dataloader):
            i += len(x)
        self.assertTrue(i == n_train)

        i = 0
        for index, (x, y) in enumerate(val_dataloader):
            i += len(x)
        self.assertTrue(i == n_val)

        i = 0
        for index, (x,y) in enumerate(test_dataloader):
            i += len(x)
        self.assertTrue(i == n_test)


if __name__ == '__main__':
    unittest.main()
