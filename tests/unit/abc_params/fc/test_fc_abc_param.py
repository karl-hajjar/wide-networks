import unittest
import os
import math
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected import ntk, ip, muP

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')

CONFIG_PATH = os.path.join(TESTS_DIR, 'resources', 'fc_abc_config.yaml')


class TestFCabcParam(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)
        self.base_model_config = ModelConfig(config_dict)
        self.width = 0

        self.ntk = ntk.FCNTK(self.base_model_config, self.width)
        self.ip = ip.FCIP(self.base_model_config, c=0, width=self.width)
        self.muP = muP.FCmuP(self.base_model_config, self.width)

    def test_ntk_scales(self):
        self.assertEqual(self.ntk.layer_scales[0], 1)
        self.assertEqual(self.ntk.init_scales[0], 1)
        self.assertEqual(self.ntk.lr_scales[0], 1)

        for l in range(1, self.ntk.n_layers):
            self.assertAlmostEqual(self.ntk.layer_scales[l], 1 / math.sqrt(self.ntk.width), places=6)
            self.assertEqual(self.ntk.init_scales[l], 1)
            self.assertEqual(self.ntk.lr_scales[l], 1)

    def test_muP_scales(self):
        self.assertAlmostEqual(self.muP.layer_scales[0], math.sqrt(self.muP.width), places=6)
        self.assertAlmostEqual(self.muP.init_scales[0], 1 / math.sqrt(self.muP.width), places=6)
        self.assertEqual(self.muP.lr_scales[0], 1)

        for l in range(1, self.muP.n_layers - 1):
            self.assertEqual(self.muP.layer_scales[l], 1)
            self.assertAlmostEqual(self.muP.init_scales[l], 1 / math.sqrt(self.muP.width), places=6)
            self.assertEqual(self.muP.lr_scales[l], 1)

        self.assertAlmostEqual(self.muP.init_scales[self.muP.n_layers-1], 1 / math.sqrt(self.muP.width), places=6)
        self.assertAlmostEqual(self.muP.init_scales[self.muP.n_layers-1], 1 / math.sqrt(self.muP.width), places=6)
        self.assertEqual(self.muP.lr_scales[self.muP.n_layers-1], 1)

    def test_ip_scales(self):
        self.assertEqual(self.ip.layer_scales[0], 1)
        self.assertEqual(self.ip.init_scales[0], 1)

        for l in range(1, self.ip.n_layers):
            self.assertAlmostEqual(self.ip.layer_scales[l], 1 / self.ip.width, places=6)
            self.assertEqual(self.ip.init_scales[l], 1)

    def test_initial_gaussian_output_ntk(self):
        n_trials = 250
        widths = [128, 256, 512, 1024]
        x = 2 * torch.rand(self.base_model_config.architecture["input_size"], requires_grad=False) - 1
        norm_2_x = torch.norm(x, 2).item()
        x = x / norm_2_x  # normalized x

        for width in widths:
            config = deepcopy(self.base_model_config)
            config.architecture["width"] = width
            config.architecture["bias"] = False
            ntk_fc = ntk.FCNTK(config, width=None)

            outputs = []
            for _ in range(n_trials):
                ntk_fc.train()
                ntk_fc.initialize_params(config.initializer)
                with torch.no_grad():
                    ntk_fc.input_layer.bias.data.fill_(0.)  # reset bias to zero
                ntk_fc.eval()
                with torch.no_grad():
                    outputs.append(ntk_fc.forward(x).detach()[0].item())
            plt.figure(figsize=(12, 6))
            sns.distplot(outputs, hist=True, kde=True)
            xs = np.arange(start=-4.0, stop=4.0, step=0.1)
            ys = (1 / np.sqrt(2 * math.pi)) * np.exp(-(xs ** 2)/2)
            plt.plot(xs, ys, label='std Gaussian')
            plt.title('NTK output at initialization with m = {:,}'.format(width))
            plt.savefig(os.path.join(FIGURES_DIR, 'ntk_init_{}.png'.format(width)))
            plt.show()

    def test_initial_vanishing_output_muP(self):
        n_trials = 200
        widths = [25, 50, 128, 256, 512, 750, 1024, 1200, 1400]
        x = 2 * torch.rand(self.base_model_config.architecture["input_size"], requires_grad=False) - 1
        norm_2_x = torch.norm(x, 2).item()
        x = x / norm_2_x  # normalized x
        results = {width: dict() for width in widths}

        results_df = pd.DataFrame(columns=['m', 'abs output', 'squared output'], dtype=float)
        idx = 0

        for width in widths:
            config = deepcopy(self.base_model_config)
            config.architecture["width"] = width
            config.architecture["bias"] = False
            muP_fc = muP.FCmuP(config, width=None)

            outputs = []
            for _ in range(n_trials):
                muP_fc.train()
                muP_fc.initialize_params(config.initializer)
                with torch.no_grad():
                    muP_fc.input_layer.bias.data.fill_(0.)  # reset bias to zero
                muP_fc.eval()
                with torch.no_grad():
                    output = muP_fc.forward(x).detach()[0].item()
                    outputs.append(output)
                    results_df.loc[idx, :] = [width, np.abs(output), output ** 2]
                    idx += 1

        plt.figure(figsize=(12, 6))
        plt.title('muP output at initialization vs m')

        sns.lineplot(data=results_df, x='m', y='abs output', label='abs output')
        plt.plot(widths, 1 / np.sqrt(widths), label='1/sqrt{m}')

        g = sns.lineplot(data=results_df, x='m', y='squared output', label='squared output')
        plt.plot(widths, 1 / np.array(widths), label='1/m')

        g.set(yscale="log")
        plt.yscale('log')

        plt.ylabel('log output')
        plt.legend()
        plt.savefig(os.path.join(FIGURES_DIR, 'muP_init.png'))
        plt.show()

    def test_initial_vanishing_output_ip(self):
        n_trials = 200
        widths = [25, 50, 128, 256, 512, 750, 1024, 1200, 1400]
        x = 2 * torch.rand(self.base_model_config.architecture["input_size"], requires_grad=False) - 1
        norm_2_x = torch.norm(x, 2).item()
        x = x / norm_2_x  # normalized x
        results = {width: dict() for width in widths}

        results_df = pd.DataFrame(columns=['m', 'abs output', 'squared output'], dtype=float)
        idx = 0

        for width in widths:
            config = deepcopy(self.base_model_config)
            config.architecture["width"] = width
            config.architecture["bias"] = False
            ip_fc = ip.FCIP(config, c=0., width=None)

            outputs = []
            for _ in range(n_trials):
                ip_fc.train()
                ip_fc.initialize_params(config.initializer)
                with torch.no_grad():
                    ip_fc.input_layer.bias.data.fill_(0.)  # reset bias to zero
                ip_fc.eval()
                with torch.no_grad():
                    output = ip_fc.forward(x).detach()[0].item()
                    outputs.append(output)
                    results_df.loc[idx, :] = [width, np.abs(output), output ** 2]
                    idx += 1

        plt.figure(figsize=(12, 6))
        plt.title('IP output at initialization vs m')

        sns.lineplot(data=results_df, x='m', y='abs output', label='abs output')
        exponent = (self.base_model_config.architecture["n_layers"] - 1)
        plt.plot(widths, np.array(widths, dtype=float) ** (-exponent / 2), label='m^(-{}/2)'.format(exponent))

        g = sns.lineplot(data=results_df, x='m', y='squared output', label='squared output')
        plt.plot(widths, np.array(widths, dtype=float) ** (-exponent), label='m^(-{})'.format(exponent))

        g.set(yscale="log")
        plt.yscale('log')

        plt.ylabel('log output')
        plt.legend()
        plt.savefig(os.path.join(FIGURES_DIR, 'ip_init.png'))
        plt.show()

    def test_copy_params_from_model(self):
        self.muP.copy_initial_params_from_model(self.ntk)
        self.ip.copy_initial_params_from_model(self.ntk)
        for l in range(self.base_model_config.architecture["n_layers"]):
            self.assertTrue((self.muP.U[l] == self.ntk.U[l]).all())
            self.assertTrue((self.ip.U[l] == self.ntk.U[l]).all())

            self.assertTrue((self.muP.v[l] == self.ntk.v[l]).all())
            self.assertTrue((self.ip.v[l] == self.ntk.v[l]).all())


if __name__ == '__main__':
    unittest.main()
