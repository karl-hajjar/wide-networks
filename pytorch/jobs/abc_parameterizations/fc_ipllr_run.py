import os
import logging
import click
from copy import deepcopy

from pytorch.job_runners.abc_parameterizations.abc_runner import ABCRunner
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
from utils.tools import read_yaml

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))  # go back 3 times from this directory
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations')
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'fc_ipllr'
CONFIG_FILE = 'fc_ipllr.yaml'

# N_TRIALS = 5
# N_TRIALS = 2
N_TRIALS = 5
# Ls = [6]  # Total depth n_layers = L + 1
Ls = [6]  # Total depth n_layers = L + 1
WIDTHS = [1024]
N_WARMUP_STEPS = 1
LR_DECAY = 0.75


@click.command()
@click.option('--activation', '-act', required=False, type=click.STRING, default="relu",
              help='Which activation function to use for the network')
@click.option('--n_steps', '-N', required=False, type=click.INT, default=300,
              help='How many steps of SGD to take')
# @click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
#               help='Which learning rate to use')
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.09,
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
    # val_dataset = deepcopy(training_dataset)  # copy train_data into validation_data
    val_dataset = deepcopy(test_dataset)  # copy test_data into validation_data to keep track of test loss

    for L in Ls:
        for width in WIDTHS:
            config_dict['architecture']['width'] = width
            config_dict['architecture']['n_layers'] = L + 1
            config_dict['optimizer']['params']['lr'] = base_lr
            config_dict['activation']['name'] = activation
            config_dict['training']['n_steps'] = n_steps
            config_dict['training']['batch_size'] = batch_size
            config_dict['scheduler'] = {'name': 'warmup_switch',
                                        'params': {'n_warmup_steps': N_WARMUP_STEPS,
                                                   'calibrate_base_lr': True,
                                                   'default_calibration': False}}
            if activation == "relu":
                config_dict['scheduler']['params']['lr_decay'] = LR_DECAY

            runner = ABCRunner(config_dict, base_experiment_path, model=FcIPLLR, train_dataset=training_dataset,
                               test_dataset=test_dataset, val_dataset=val_dataset, early_stopping=False,
                               n_trials=N_TRIALS, calibrate_base_lr=True)
            runner.run(n_warmup_steps=N_WARMUP_STEPS)


if __name__ == '__main__':
    run()
