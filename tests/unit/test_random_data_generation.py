import unittest
import numpy as np

from utils.data.random import RandomData

PRECISION = 1.0e-6


class TestRandomDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.rs = [0.5, 1, 2]
        self.ks = [2, 3, 4, 5, 10, 15]
        self.ds = [2, 3, 4, 5, 10, 20, 100, 1000]
        self.ns = [10, 50, 100, 500, 1000]

    def test_cluster_centers(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    data = RandomData(k=k, r=r, d=d)
                    self.assertTrue(len(data.cluster_centers_) == 0)

                    cluster_centers = data.cluster_centers
                    self.assertTrue(len(cluster_centers.shape) == 2)
                    self.assertSequenceEqual(cluster_centers.shape, (k ** 2, d))
                    self.assertTrue((cluster_centers >= (-r + data.B)).all())
                    self.assertTrue((cluster_centers <= PRECISION + (r - data.B)).all())

                    self.check_distances_(cluster_centers, 3 * data.B, r)

    def check_distances_(self, cluster_centers, dist, r):
        min_dist = 2 * r
        for i in range(len(cluster_centers) - 1):
            distances = np.linalg.norm(cluster_centers[i+1:, :] - cluster_centers[i, :], axis=1)
            min_dist = min(min_dist, np.min(distances))
        self.assertTrue(min_dist >= dist - PRECISION)

    def test_uniform_ball_samples(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = RandomData(k=k, r=r, d=d)
                        samples = data.uniform_sample_ball_(n)
                        self.assertSequenceEqual(samples.shape, (n, d))
                        self.assertTrue((np.linalg.norm(samples, axis=1) < data.B + PRECISION).all())

    def test_sample_generation_with_n(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = RandomData(k=k, r=r, d=d)
                        x, cluster_assignments, y = data.generate_samples(n)
                        sample_centers = np.array([data.cluster_centers[cluster_assignments[i]] for i in range(n)])

                        # x
                        self.assertSequenceEqual(x.shape, (n, d))
                        self.assertTrue((-r - PRECISION < x).all())
                        self.assertTrue((x < r + PRECISION).all())
                        self.assertTrue((np.linalg.norm(x - sample_centers, axis=1) < data.B + PRECISION).all())

                        # cluster assignments
                        self.assertSequenceEqual(cluster_assignments.shape, (n,))
                        self.assertTrue((0 <= cluster_assignments).all())
                        self.assertTrue((cluster_assignments < k ** 2).all())
                        self.assertTrue(np.sum((data.cluster_labels + 1) / 2) /
                                        len(data.cluster_labels) >= 0.5 - data.dev)
                        self.assertTrue(np.sum((data.cluster_labels + 1) / 2) /
                                        len(data.cluster_labels) <= 0.5 + data.dev)

                        # y
                        self.assertSequenceEqual(y.shape, (n,))
                        self.assertTrue((y == 1).sum() + (y == -1).sum() == len(y))

                        if n >= 1000:
                            if np.sum((y + 1) / 2) / len(y) < 0.2:
                                print(np.sum((y + 1) / 2) / len(y))
                            self.assertTrue(np.sum((y + 1) / 2) / len(y) >= 0.3)
                            self.assertTrue(np.sum((y + 1) / 2) / len(y) <= 0.7)

    def test_sample_generation_without_n(self):
        for r in self.rs:
            for k in self.ks:
                for d in self.ds:
                    for n in self.ns:
                        data = RandomData(k=k, r=r, d=d, n=n)
                        x, cluster_assignments, y = data.generate_samples()
                        sample_centers = np.array([data.cluster_centers[cluster_assignments[i]] for i in range(n)])

                        # x
                        self.assertSequenceEqual(x.shape, (n, d))
                        self.assertTrue((-r - PRECISION < x).all())
                        self.assertTrue((x < r + PRECISION).all())
                        self.assertTrue((np.linalg.norm(x - sample_centers, axis=1) < data.B + PRECISION).all())

                        # cluster assignments
                        self.assertSequenceEqual(cluster_assignments.shape, (n,))
                        self.assertTrue((0 <= cluster_assignments).all())
                        self.assertTrue((cluster_assignments < k ** 2).all())
                        self.assertTrue(np.sum((data.cluster_labels + 1) / 2) /
                                        len(data.cluster_labels) >= 0.5 - data.dev)
                        self.assertTrue(np.sum((data.cluster_labels + 1) / 2) /
                                        len(data.cluster_labels) <= 0.5 + data.dev)

                        # y
                        self.assertSequenceEqual(y.shape, (n,))
                        self.assertTrue((y == 1).sum() + (y == -1).sum() == len(y))

                        if n >= 1000:
                            if np.sum((y + 1) / 2) / len(y) < 0.2:
                                print(np.sum((y + 1) / 2) / len(y))
                            self.assertTrue(np.sum((y + 1) / 2) / len(y) >= 0.3)
                            self.assertTrue(np.sum((y + 1) / 2) / len(y) <= 0.7)


if __name__ == '__main__':
    unittest.main()
