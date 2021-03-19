import os
import click

from pytorch.job_runners.abc_parameterizations.abc_runner import ABCRunner
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.tools import read_yaml
from utils.data.mnist import load_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times in the dir from this directory
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'standard_fc_ip_mnist'
CONFIG_FILE = 'standard_fc_ip_mnist.yaml'


@click.option('-N', '--n_trials', required=False, type=click.INT,
              help='Number of different random initializations')
@click.option('-dld', '--download', required=False, type=click.BOOL,
              help='Number of different random initializations')
def run(n_trials=10, download=False):
    # print('FILE_DIR :', FILE_DIR)
    # print('os.path.dirname(FILE_DIR) :', os.path.dirname(FILE_DIR))
    # print('os.path.dirname(os.path.dirname(FILE_DIR)) :', os.path.dirname(os.path.dirname(FILE_DIR)))
    # print('ROOT :', ROOT)
    config_dict = read_yaml(os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', CONFIG_FILE))

    # define corresponding directory in experiments folder
    base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, MODEL_NAME)  # base experiment folder

    # prepare data
    training_dataset, test_dataset = load_data(download=download, flatten=True)

    runner = ABCRunner(config_dict, base_experiment_path, model=StandardFCIP, train_dataset=training_dataset,
                       test_dataset=test_dataset, early_stopping=False, n_trials=n_trials)
    runner.run()


if __name__ == '__main__':
    run()
