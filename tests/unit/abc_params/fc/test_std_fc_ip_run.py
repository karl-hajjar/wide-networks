import unittest
import os

from pytorch.job_runners.abc_parameterizations.abc_runner import ABCRunner
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.tools import read_yaml
from utils.data.mnist import load_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR))))  # go back 4 times
EXPERIMENTS_DIR = 'experiments'
MODEL_NAME = 'standard_fc_ip_mnist'
CONFIG_FILE = 'standard_fc_ip_mnist.yaml'

# Run
# ```tensorboard --logdir path_to_the_exp_version```
# Alternatively, one can run
# tensorboard --logdir_spec=name1:./bsize=1_ntrain=64_d=256_m=100,name2:./bsize=1_ntrain=64_d=256_m=1000
# to specicfy multiple log directories
# in the command line to and then open http://localhost:6006/ to see the TensorBoard


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        n_trials = 4
        n_epochs = 5
        config_dict = read_yaml(os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', CONFIG_FILE))
        config_dict['training']['n_epochs'] = n_epochs
        config_dict['training']['n_steps'] = 5000

        # define corresponding directory in experiments folder
        base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, MODEL_NAME)  # base experiment folder

        # prepare data
        training_dataset, test_dataset = load_data(download=False, flatten=True)

        self.runner = ABCRunner(config_dict, base_experiment_path, model=StandardFCIP, train_dataset=training_dataset,
                                test_dataset=test_dataset, early_stopping=False, n_trials=n_trials)

    def test_something(self):
        print('Hello')
        self.assertEqual(True, True)

    def test_run(self):
        self.runner.run()


if __name__ == '__main__':
    unittest.main()
