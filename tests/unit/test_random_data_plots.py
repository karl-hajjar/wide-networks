import unittest

from utils.dataset.random import RandomData
from utils.plot.data import *
# export PYTHONPATH=$PYTHONPATH:'/Users/khajjar/Documents/projects/dl-frameworks-mnist/'


class TestRandomDataPlots(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_2d_generation_plot(self):
        k, r, n, d = 3, 1, 100, 2
        data = RandomData(k, r, n, d)
        self.assertTrue(len(data.cluster_centers_) == 0)

        cluster_centers = data.cluster_centers
        self.assertTrue(len(data.cluster_centers_) == k ** 2)
        self.assertTrue((data.cluster_centers_ <= (r - data.B)).all())
        self.assertTrue((data.cluster_centers_ >= (-r + data.B)).all())

        plot_cluster_centers2d(cluster_centers)
        plt.savefig('outputs/figures/2d_test_fig.png')
        plt.show()

        plot_cluster_centers2d(cluster_centers, show_grid=False)
        plt.show()

        plot_cluster_centers2d(cluster_centers, show_ticks=False)
        plt.show()

        plot_cluster_centers2d(cluster_centers, show_grid=False, show_ticks=False)
        plt.show()

    def test_2d_mutiple_params(self):
        rs = [0.5, 1, 2]
        ks = [2, 3, 4, 10, 15]
        ds = [2, 3, 4, 10, 20]

        for r in rs:
            for k in ks:
                for d in ds:
                    data = RandomData(k=k, r=r, d=d)
                    plot_cluster_centers2d(data.cluster_centers, style=None, color=None)
                    plt.savefig('outputs/figures/2d_test_fig_r={}_k={}_d={}.png'.format(r, k, d))
                    plt.show()

    def test_3d_generation_plot(self):
        k, r, n, d = 3, 1, 100, 3
        data = RandomData(k, r, n, d)
        self.assertTrue(len(data.cluster_centers_) == 0)

        cluster_centers = data.cluster_centers
        self.assertTrue(len(data.cluster_centers) == k ** 2)
        self.assertTrue(len(data.cluster_centers.shape) == 2)
        self.assertSequenceEqual(data.cluster_centers.shape, (k ** 2, d))

        plot_cluster_centers3d(cluster_centers)
        plt.savefig('outputs/figures/3d_test_fig.png')
        plt.show()

        plot_cluster_centers3d(cluster_centers, show_grid=False)
        plt.show()

        plot_cluster_centers3d(cluster_centers, show_ticks=False)
        plt.show()

    def test_3d_mutiple_params(self):
        rs = [0.5, 1, 2]
        ks = [2, 3, 4, 10, 15]
        ds = [3, 4, 10, 20]

        for r in rs:
            for k in ks:
                for d in ds:
                    data = RandomData(k=k, r=r, d=d)
                    plot_cluster_centers3d(data.cluster_centers)
                    plt.savefig('outputs/figures/3d_test_fig_r={}_k={}_d={}.png'.format(r, k, d))
                    plt.show()

    def test_uniform_ball_2d(self):
        rs = [0.5, 1, 2]
        ks = [2, 3, 4, 10, 15]
        d = 2
        ns = [10, 50, 100, 500, 1000]

        for r in rs:
            for k in ks:
                for n in ns:
                    data = RandomData(k=k, r=r, d=d)
                    samples = data.uniform_sample_ball_(n)
                    B = data.B

                    plot_cluster_centers2d(samples, style=None, color=None)
                    ax = plt.gca()
                    draw_circle = plt.Circle((0., 0.), B, fill=False, color='g')
                    ax.add_artist(draw_circle)
                    ax.set(xlim=(-B, B), ylim=(-B, B))
                    plt.savefig('outputs/figures/2d_test_unif_ball={}_k={}_d={}.png'.format(r, k, d))
                    plt.show()

    def test_uniform_ball_2d_mutiple_d(self):
        rs = [1, 2]
        ks = [3, 4, 10]
        ds = [2, 3, 5, 10, 50, 100]
        ns = [100, 500, 1000]

        for r in rs:
            for k in ks:
                for n in ns:
                    for d in ds:
                        data = RandomData(k=k, r=r, d=d)
                        samples = data.uniform_sample_ball_(n)
                        B = data.B

                        plot_cluster_centers2d(samples, style=None, color=None)
                        ax = plt.gca()
                        draw_circle = plt.Circle((0., 0.), B, fill=False, color='g')
                        ax.add_artist(draw_circle)
                        ax.set(xlim=(-B, B), ylim=(-B, B))
                        plt.savefig('outputs/figures/2d_test_unif_ball={}_k={}_d={}.png'.format(r, k, d))
                        plt.show()

    def test_uniform_ball_3d(self):
        rs = [0.5, 1, 2]
        ks = [2, 3, 4, 10, 15]
        d = 3
        ns = [10, 50, 100, 500, 1000]

        for r in rs:
            for k in ks:
                for n in ns:
                    data = RandomData(k=k, r=r, d=d)
                    samples = data.uniform_sample_ball_(n)
                    plot_cluster_centers3d(samples, style=None, color=None)
                    plt.savefig('outputs/figures/3d_test_unif_ball={}_k={}_d={}.png'.format(r, k, d))
                    plt.show()

    def test_random_data_plot(self):
        rs = [1, 2]
        ks = [3, 4, 10]
        ds = [2, 3, 5, 10, 50, 100]
        ns = [100, 500, 1000]

        for r in rs:
            for k in ks:
                for n in ns:
                    for d in ds:
                        data = RandomData(k=k, r=r, d=d)
                        x, cluster_assignments, y = data.generate_samples(n)
                        sample_centers = np.array([data.cluster_centers[cluster_assignments[i]] for i in range(n)])

                        plot_random_data2d(x, sample_centers, y, style=None, color='k', show_grid=False)
                        plt.savefig('outputs/figures/2d_test_random_plot_r={}_k={}_d={}.png'.format(r, k, d))
                        plt.show()


if __name__ == '__main__':
    unittest.main()
