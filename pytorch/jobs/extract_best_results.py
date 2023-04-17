import os
import logging
import click
import numpy as np

from utils.tools import load_pickle


FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times from this directory
EXPERIMENTS_DIR = 'experiments'

ACTIVATIONS = ['relu', 'gelu', 'elu', 'tanh']
BEST_RESULTS_FILE = 'best_results.pickle'


@click.command()
@click.option('--depth', '-L', required=False, type=click.INT, default=5,
              help='The `L` used in the experiment')
@click.option('--width', '-m', required=False, type=click.INT, default=1024,
              help='The width `m` used in the experiment')
@click.option('--dataset', '-ds', required=False, type=click.STRING, default="mnist",
              help='Which dataset was used in the experiment')
@click.option('--n_trials', '-N', required=False, type=click.INT, default=5,
              help='The number of trials in the experiment')
@click.option('--model', '-mod', required=False, type=click.STRING, default="ipllr",
              help='Which model was used in the experiment')
def run(depth=5, width=1024, n_trials=5, dataset="mnist", model="ipllr"):
    logger = logging.getLogger()

    base_experiments_dir = os.path.join(ROOT, EXPERIMENTS_DIR, 'fc_{}_{}', 'L={}_m={}'.format(model, dataset, depth,
                                                                                              width))
    if not os.path.exists(base_experiments_dir):
        logger.warning("Path {} does not exits, stopping results extraction.".format(base_experiments_dir))
    else:
        logger.info("Retrieving results from directory : {}".format(base_experiments_dir))
        test_accuracy_by_lr_dict = dict()
        for activation in ACTIVATIONS:
            test_accuracy_by_lr_dict[activation] = dict()
            for f in os.scandir(base_experiments_dir):
                if activation in f.name:
                    lr = float(f.name.split('lr=')[1].split('_')[0])
                    accuracies = []
                    for i in range(1, n_trials + 1):
                        best_results_path = os.path.join(base_experiments_dir, f.name, 'trial_{}'.format(i),
                                                         BEST_RESULTS_FILE)
                        accuracies.append(load_pickle(best_results_path, single=True)['accuracy'])
                    test_accuracy_by_lr_dict[activation][lr] = np.mean(accuracies)
                    best_lr = max(test_accuracy_by_lr_dict, key=test_accuracy_by_lr_dict.get)
                    logger.info("For activation {}: best lr = {} with accuracy = {:.5f}".
                                format(activation, best_lr, test_accuracy_by_lr_dict[best_lr]))


if __name__ == '__main__':
    run()
