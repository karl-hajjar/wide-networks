import os
import logging
import click
from copy import deepcopy

from pytorch.job_runners.abc_parameterizations.abc_runner import ABCRunner
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.tools import read_yaml

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))  # go back 3 times from this directory
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations')
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'standard_fc_ip'
CONFIG_FILE = 'standard_fc_ip.yaml'

N_TRIALS = 5
Ls = [6]  # n_layers - 1
WIDTHS = [1024]


@click.command()
@click.option('--activation', '-act', required=False, type=click.STRING, default="relu",
              help='Which activation function to use for the network')
@click.option('--n_steps', '-N', required=False, type=click.INT, default=300,
              help='How many steps of SGD to take')
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
              help='Which learning rate to use')
@click.option('--batch_size', '-bs', required=False, type=click.INT, default=512,
              help='What batch size to use')
@click.option('--dataset', '-ds', required=False, type=click.STRING, default="mnist",
              help='Which dataset to train on')
@click.option('--download', '-dld', required=False, type=click.BOOL, default=False,
              help='Whether to download the data or not')
def run(activation="relu", n_steps=300, base_lr=0.01, batch_size=512, dataset="mnist", download=False):
    model_name = '{}_{}'.format(MODEL_NAME, dataset)
    config_path = os.path.join(CONFIG_PATH, '{}.yaml'.format(model_name))
    config_dict = read_yaml(config_path)

    # define corresponding directory in experiments folder
    base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, model_name)  # base experiment folder

    # Load data & define models
    logger = logging.getLogger()
    logger.info('Loading data ...')
    if dataset == 'mnist':
        from utils.dataset.mnist import load_data
    elif dataset == 'cifar10':
        from utils.dataset.cifar10 import load_data
    elif dataset == 'cifar100':
        # TODO : add cifar100 to utils.dataset
        pass
    else:
        error = ValueError("dataset must be one of ['mnist', 'cifar10', 'cifar100'] but was {}".format(dataset))
        logger.error(error)
        raise error

    # prepare data
    training_dataset, test_dataset = load_data(download=download, flatten=True)
    val_dataset = deepcopy(training_dataset)  # copy train_data into validation_data

    for L in Ls:
        for width in WIDTHS:
            config_dict['architecture']['width'] = width
            config_dict['architecture']['n_layers'] = L + 1
            config_dict['optimizer']['params']['lr'] = base_lr
            config_dict['activation']['name'] = activation
            config_dict['training']['n_steps'] = n_steps
            config_dict['training']['batch_size'] = batch_size
            runner = ABCRunner(config_dict, base_experiment_path, model=StandardFCIP, train_dataset=training_dataset,
                               test_dataset=test_dataset, val_dataset=val_dataset, early_stopping=False,
                               n_trials=N_TRIALS)
            runner.run()


if __name__ == '__main__':
    run()
