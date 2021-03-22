import os
import click

from pytorch.job_runners.abc_parameterizations.abc_runner import ABCRunner
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.tools import read_yaml
from utils.data.mnist import load_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))  # go back 3 times from this directory
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'standard_fc_ip_mnist'
CONFIG_FILE = 'standard_fc_ip_mnist.yaml'

# Ls = [2, 3]  # n_layers - 1
# WIDTHS = [128, 256, 512, 1024, 1400]
Ls = [2]  # n_layers - 1
WIDTHS = [128]
# Ls = [4, 5]  # n_layers - 1
# WIDTHS = [128, 256, 512, 1024]


@click.command()
@click.option('--n_trials', '-N', required=False, type=click.INT, default=10,
              help='Number of different random initializations')
@click.option('--download', '-dld', required=False, type=click.BOOL, default=False,
              help='Whether to download the data or not')
def run(n_trials=10, download=False):
    config_dict = read_yaml(os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', CONFIG_FILE))

    # define corresponding directory in experiments folder
    base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, MODEL_NAME)  # base experiment folder

    # prepare data
    training_dataset, test_dataset = load_data(download=download, flatten=True)

    for L in Ls:
        for width in WIDTHS:
            config_dict['architecture']['n_layers'] = L + 1
            config_dict['architecture']['width'] = width
            runner = ABCRunner(config_dict, base_experiment_path, model=StandardFCIP, train_dataset=training_dataset,
                               test_dataset=test_dataset, early_stopping=False, n_trials=n_trials)
            runner.run()


if __name__ == '__main__':
    run()
