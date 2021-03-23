import unittest
import os
from copy import deepcopy

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')

CONFIG_PATH = os.path.join(TESTS_DIR, 'resources', 'fc_abc_config.yaml')


class TestFcStandardIP(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)
        self.base_model_config = ModelConfig(config_dict)
        self.width = 0

        self.standard_ip = StandardFCIP(self.base_model_config, self.width)

    def test_scales(self):
        Ls = [2, 3, 4, 5, 10]
        for L in Ls:
            model_config = deepcopy(self.base_model_config)
            model_config.architecture['n_layers'] = L + 1
            standard_ip = StandardFCIP(model_config, self.width)

            self.assertEqual(standard_ip.n_layers, L+1)

            self.assertEqual(standard_ip.layer_scales[0], 1)
            self.assertEqual(standard_ip.init_scales[0], 1)
            self.assertEqual(standard_ip.lr_scales[0], standard_ip.width)

            for l in range(1, standard_ip.n_layers - 1):
                self.assertAlmostEqual(standard_ip.layer_scales[l], 1 / standard_ip.width, places=6)
                self.assertEqual(standard_ip.init_scales[l], 1)
                self.assertEqual(standard_ip.lr_scales[l], standard_ip.width ** 2)

            self.assertAlmostEqual(standard_ip.layer_scales[standard_ip.n_layers-1], 1 / standard_ip.width, places=6)
            self.assertEqual(standard_ip.init_scales[standard_ip.n_layers-1], 1)
            self.assertEqual(standard_ip.lr_scales[standard_ip.n_layers-1], standard_ip.width)

    def test_ip_param_groups_lr(self):
        Ls = [2, 3, 4, 5, 10]
        for L in Ls:
            model_config = deepcopy(self.base_model_config)
            model_config.architecture['n_layers'] = L + 1
            standard_ip = StandardFCIP(model_config, self.width)

            self.assertEqual(len(standard_ip.optimizer.param_groups), standard_ip.n_layers)
            lrs_from_model = standard_ip._get_opt_lr()
            lrs_from_opt = [param_group['lr'] for param_group in standard_ip.optimizer.param_groups]
            self.assertEqual(lrs_from_model[0], lrs_from_opt[0])
            for l in range(1, standard_ip.n_layers - 1):
                self.assertEqual(lrs_from_model[1], lrs_from_opt[l])
            self.assertEqual(lrs_from_model[2], lrs_from_opt[standard_ip.n_layers - 1])


if __name__ == '__main__':
    unittest.main()
