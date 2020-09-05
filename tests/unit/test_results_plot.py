import unittest
import os
import pickle

from utils.plot.metrics import *

SAVE_DIR = '../../experiments'
NAME = 'wide_2_layer'
RESULTS_FILE = 'results.pickle'

Ds = [10, 20, 25, 50, 128]
Ms = [10, 50, 100, 200, 500, 1000]

# Ds = [5, 10, 15, 20, 25, 50, 128, 256, 512, 1000, 5000, 10000]
# Ms = [10, 50, 100, 200, 500, 1000, 2000, 5000]


class TestResultsPlots(unittest.TestCase):
    def setUp(self) -> None:
        base_experiment_name = 'activation=relu_loss=logistic_opt=sgd_init=custom'
        k_r_config = 'k=3_r=0.5'
        self.experiments_dir = os.path.join(SAVE_DIR, NAME, base_experiment_name, k_r_config)

        self.batch_size, self.n_train = 1, 64
        self.experiments_results = dict()
        self.n_trials = 5

    def _read_results(self, exp_name):
        self.experiments_results[exp_name] = []
        for i in range(1, self.n_trials + 1):
            with open(os.path.join(self.experiments_dir, exp_name, 'trial_' + str(i), RESULTS_FILE), 'rb') as file:
                self.experiments_results[exp_name].append(pickle.load(file))

    def test_metric_vs_other_plot(self):
        d = 50
        y_metric_key = 'normalized_margin'
        x_metric = []
        y_metric = []
        for m in Ms:
            exp_name = 'bsize={}_ntrain={}_d={}_m={}'.format(self.batch_size, self.n_train, d, m)
            x_metric.append(m)
            if exp_name not in self.experiments_results.keys():
                self._read_results(exp_name)
            y_metric.append([self.experiments_results[exp_name][i]['test'][0][y_metric_key]
                             for i in range(self.n_trials)])

        plot_metric_vs_other(x_metric, y_metric, linewidth=2.0, title='normalized margin vs m with d={:,}'.format(d),
                             xlabel='m', ylabel='normalized margin')
        plt.savefig('outputs/figures/bsize={}_ntrain={}_d={}_{}_vs_m.png'.format(self.batch_size, self.n_train, d,
                                                                                 y_metric_key))
        plt.show()


if __name__ == '__main__':
    unittest.main()
