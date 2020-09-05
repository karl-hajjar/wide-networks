import os
import click

from pytorch.job_runners.wide_2_layer_runner import Wide2LayerRunner
from utils.tools import *

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(FILE_DIR))  # go back 2 times in the dir from this directory
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'wide_2_layer'
CONFIG_FILE = 'wide_two_layer_net.yaml'


@click.option('-N', '--n_rep', required=False, type=click.INT,
              help='Number of times that each experiment will be repeated with different random seeds')
def run(n_rep=5):
    print('FILE_DIR :', FILE_DIR)
    print('os.path.dirname(FILE_DIR) :', os.path.dirname(FILE_DIR))
    print('os.path.dirname(os.path.dirname(FILE_DIR)) :', os.path.dirname(os.path.dirname(FILE_DIR)))
    print('ROOT :', ROOT)
    config_dict = read_yaml(os.path.join(ROOT, 'pytorch/configs/', CONFIG_FILE))  # read model config

    # define corresponding directory in experiments folder
    base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, MODEL_NAME)  # base experiment folder

    runner = Wide2LayerRunner(config_dict, base_experiment_path, n_rep)
    runner.run()


if __name__ == '__main__':
    run()
