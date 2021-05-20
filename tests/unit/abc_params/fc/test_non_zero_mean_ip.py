import unittest
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')
CONFIG_PATH = os.path.join(TESTS_DIR, 'resources', 'fc_abc_config.yaml')


class TestNonCenteredIP(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)
        L = 4
        width = 1024
        mean = 1.0

        self.input_size = config_dict['architecture']['input_size']
        self.base_lr = config_dict['optimizer']['params']['lr']
        self.n_warmup_steps = 1
        self.width = width
        config_dict['architecture']['width'] = width
        self.L = L
        self.mean = mean
        config_dict['architecture']['n_layers'] = L + 1
        config_dict['optimizer']['params']['lr'] = self.base_lr

        self.base_model_config = ModelConfig(config_dict)
        self.base_model_config.initializer.params["mean"] = mean
        self.ip_non_centered = StandardFCIP(self.base_model_config)

    def test_init_mean_U(self):
        self.assertEqual(self.ip_non_centered.init_mean, self.mean)

        std = self.ip_non_centered.std
        for l in range(self.ip_non_centered.n_layers):
            if l == self.L:
                std = 1.0
            if l == 0:
                mean = 0.
            else:
                mean = self.mean
            self.plot_parameter_values_histogram(self.ip_non_centered.U[l].flatten() - mean, std,
                                                 'Distribution of the entries of (U^{} - {})'.format(l, mean))

    def test_init_mean_w(self):
        self.assertEqual(self.ip_non_centered.init_mean, self.mean)

        std = self.ip_non_centered.std

        self.plot_parameter_values_histogram(self.ip_non_centered.input_layer.weight.detach().flatten(),
                                             std, 'Distribution of the entries of (w^{}(0) - {})'.format(1, 0))
        for l, layer in enumerate(self.ip_non_centered.intermediate_layers):
            self.plot_parameter_values_histogram(layer.weight.detach().flatten() - self.mean, std,
                                                 'Distribution of the entries of (w^{}(0) - {})'.format(l+2, self.mean))
        self.plot_parameter_values_histogram(self.ip_non_centered.output_layer.weight.detach().flatten() - self.mean,
                                             1.0, 'Distribution of the entries of (w^{}(0) - {})'.\
                                             format(self.ip_non_centered.n_layers, self.mean))

    @staticmethod
    def plot_parameter_values_histogram(values: torch.Tensor, std: float, title: str):
        var = std ** 2
        plt.figure(figsize=(12, 6))

        xs = np.arange(start=-3 * var, stop=3 * var, step=0.02)
        ys = (1 / np.sqrt(2 * math.pi * var)) * np.exp(-(xs ** 2) / (2 * var))
        with torch.no_grad():
            sns.histplot(values.detach().numpy(), kde=True, stat='density')
        plt.plot(xs, ys, label='N(0,{:.0f})'.format(var), c='r')
        plt.legend()
        plt.xlabel('values')
        plt.ylabel('values density')
        plt.title(title)
        plt.show()

    def test_forward_pass_decomposition(self):
        x = 2 * torch.rand((10, self.base_model_config.architecture["input_size"]), requires_grad=False)
        m = self.width
        a = self.ip_non_centered.init_mean

        centered_U = [self.ip_non_centered.U[0]] + [U - self.mean for U in self.ip_non_centered.U[1:]]

        x = self.ip_non_centered.activation(self.ip_non_centered.input_layer.forward(x))

        for l, layer in enumerate(self.ip_non_centered.intermediate_layers):
            mean_x = a * torch.sum(x, dim=1) / m
            centered_h = F.linear(x, centered_U[l+1]) / m

            h = layer.forward(x) / m
            # use view to operate under the proper broadcasting rules of pytorch
            torch.testing.assert_allclose(h, centered_h + mean_x.view(-1, 1), atol=1e-2, rtol=1e-2)

            x = self.ip_non_centered.activation(h)

        mean_x = a * torch.sum(x, dim=1) / m
        centered_h = F.linear(x, centered_U[self.ip_non_centered.n_layers-1]) / m

        h = self.ip_non_centered.output_layer.forward(x) / m
        # use view to operate under the proper broadcasting rules of pytorch
        torch.testing.assert_allclose(h, centered_h + mean_x.view(-1, 1), atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
