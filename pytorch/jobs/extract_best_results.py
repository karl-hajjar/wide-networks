import os
import logging
import click
import numpy as np

from utils.tools import load_pickle, set_up_logger


FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times from this directory
EXPERIMENTS_DIR = 'experiments'

ACTIVATIONS = ['relu', 'gelu', 'elu', 'tanh']
# ACTIVATIONS = ['relu']
BEST_RESULTS_FILE = 'best_results.pickle'


@click.command()
@click.option('--depth', '-L', required=False, type=click.INT, default=5,
              help='The `L` used in the experiment')
@click.option('--width', '-m', required=False, type=click.INT, default=1024,
              help='The width `m` used in the experiment')
@click.option('--n_trials', '-N', required=False, type=click.INT, default=5,
              help='The number of trials in the experiment')
@click.option('--dataset', '-ds', required=False, type=click.STRING, default="mnist",
              help='Which dataset was used in the experiment')
@click.option('--model', '-mod', required=False, type=click.STRING, default="ipllr",
              help='Which model was used in the experiment')
def run(depth=5, width=1024, n_trials=5, dataset="mnist", model="ipllr"):
    print(ROOT)
    set_up_logger(os.path.join(ROOT, 'extract.log'))
    logger = logging.getLogger()

    if 'fc' not in model:
        model_name = 'fc_{}_{}'.format(model, dataset)
    else:
        model_name = '{}_{}'.format(model, dataset)

    base_experiments_dir = os.path.join(ROOT, EXPERIMENTS_DIR, model_name, 'L={}_m={}').format(depth, width)
    if not os.path.exists(base_experiments_dir):
        logger.warning("Path {} does not exist, stopping results extraction.".format(base_experiments_dir))
    else:
        try:
            logger.info("Retrieving results from directory : {}".format(base_experiments_dir))
            test_accuracy_by_lr_dict = dict()
            for activation in ACTIVATIONS:
                test_accuracy_by_lr_dict[activation] = dict()
                for f in os.scandir(base_experiments_dir):
                    file_act = f.name.split('activation=')[1].split('_')[0]
                    if file_act == activation:
                        lr = float(f.name.split('lr=')[1].split('_')[0])
                        accuracies = []
                        for i in range(1, n_trials + 1):
                            best_results_path = os.path.join(base_experiments_dir, f.name, 'trial_{}'.format(i),
                                                             BEST_RESULTS_FILE)
                            if os.path.exists(best_results_path):
                                best_results = load_pickle(best_results_path, single=True)
                                accuracies.append(best_results['test'][0]['accuracy'])
                                test_accuracy_by_lr_dict[activation][lr] = np.mean(accuracies)

                best_lr = max(test_accuracy_by_lr_dict[activation], key=test_accuracy_by_lr_dict[activation].get)
                logger.info("For activation {}: best lr = {} with accuracy = {:.5f}".
                            format(activation, best_lr, test_accuracy_by_lr_dict[activation][best_lr]))
        except Exception as e:
            logger.exception("Exception while running the best results extraction : {}".format(e))
            raise Exception(e)


if __name__ == '__main__':
    run()
