import unittest
import os

from utils.tools import read_yaml
from pytorch.jobs.wide_2_layer_run import ROOT, EXPERIMENTS_DIR, MODEL_NAME, CONFIG_FILE
from pytorch.job_runners import wide_2_layer_runner


class TestWide2LayerRunner(unittest.TestCase):
    def setUp(self) -> None:
        wide_2_layer_runner.Ks = [3, 4]
        wide_2_layer_runner.Rs = [0.5, 1]
        wide_2_layer_runner.BATCH_SIZES = [32, 128]
        wide_2_layer_runner.N_TRAINs = [256, 512]
        wide_2_layer_runner.Ds = [15, 50]
        wide_2_layer_runner.Ms = [50, 100]

        n_rep = 3

        config_dict = read_yaml(os.path.join(ROOT, 'pytorch/configs/', CONFIG_FILE))  # read model config
        self.base_experiment_path = os.path.join(ROOT, EXPERIMENTS_DIR, MODEL_NAME)

        self.runner = wide_2_layer_runner.Wide2LayerRunner(config_dict, self.base_experiment_path, n_rep)

    def test_root_and_path(self):
        # os.path.join doesn't append '/' at the end of the path
        self.assertTrue((ROOT == '/Users/khajjar/Documents/projects/dl-frameworks-mnist'))
        self.assertTrue(self.base_experiment_path ==
                        '/Users/khajjar/Documents/projects/dl-frameworks-mnist/experiments/wide_2_layer')

    def test_run(self):
        self.runner.run()


if __name__ == '__main__':
    unittest.main()
