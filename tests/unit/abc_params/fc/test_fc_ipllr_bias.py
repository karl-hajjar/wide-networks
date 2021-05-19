import unittest
import os
from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.ipllr_bias import FcIPLLRBias
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')

CONFIG_PATH = os.path.join(TESTS_DIR, 'resources', 'fc_abc_config.yaml')


class TestFcIPLLRBias(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)
        L = 4
        width = 1024

        self.input_size = config_dict['architecture']['input_size']
        self.base_lr = config_dict['optimizer']['params']['lr']
        self.n_warmup_steps = 1
        self.width = width
        config_dict['architecture']['width'] = width
        self.L = L
        config_dict['architecture']['n_layers'] = L + 1
        config_dict['optimizer']['params']['lr'] = self.base_lr
        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': self.n_warmup_steps,
                                               'calibrate_base_lr': False}}

        self.base_model_config = ModelConfig(config_dict)
        self.ipllr_bias = FcIPLLRBias(self.base_model_config, n_warmup_steps=4)

    def test_bias_attributes(self):
        self.assertTrue(self.ipllr_bias.bias)
        self.assertFalse(self.ipllr_bias.scale_bias)
        self.assertTrue(hasattr(self.ipllr_bias.input_layer, "bias"))
        for layer in self.ipllr_bias.intermediate_layers:
            self.assertTrue(hasattr(layer, "bias"))
        self.assertTrue(hasattr(self.ipllr_bias.output_layer, "bias"))

    def test_bias_vs_ipllr(self):
        ipllr = FcIPLLR(self.base_model_config, n_warmup_steps=1)
        ipllr.copy_initial_params_from_model(self.ipllr_bias)
        ipllr.initialize_params()
        with torch.no_grad():
            torch.testing.assert_allclose(self.ipllr_bias.input_layer.bias.detach().data,
                                          ipllr.input_layer.bias.detach().data, atol=1e-5, rtol=1e-5)
            for l, layer in enumerate(self.ipllr_bias.intermediate_layers):
                ipllr_layer = getattr(ipllr.intermediate_layers, "layer_{:,}_intermediate".format(l+2))
                torch.testing.assert_allclose(layer.bias.detach().data,
                                              ipllr_layer.bias.detach().data, atol=1e-5, rtol=1e-5)
            torch.testing.assert_allclose(self.ipllr_bias.output_layer.bias.detach().data,
                                          ipllr.output_layer.bias.detach().data, atol=1e-5, rtol=1e-5)

    def test_biases_init_dist(self):
        std = self.ipllr_bias.std
        self.plot_parameter_values_histogram(self.ipllr_bias.input_layer.bias, std,
                                             'Distribution of parameter values at layer 1')
        for l, layer in enumerate(self.ipllr_bias.intermediate_layers):
            self.plot_parameter_values_histogram(layer.bias, std,
                                                 'Distribution of parameter values at layer {:,}'.format(l+2))
        self.plot_parameter_values_histogram(self.ipllr_bias.output_layer.bias, 1.0,
                                             'Distribution of parameter values at layer {:,}'.\
                                             format(self.ipllr_bias.n_layers))

    @staticmethod
    def plot_parameter_values_histogram(values: torch.Tensor, std: float, title: str):
        var = std ** 2
        plt.figure(figsize=(12, 6))

        xs = np.arange(start=-2 * var, stop=2 * var, step=0.02)
        ys = (1 / np.sqrt(2 * math.pi * var)) * np.exp(-(xs ** 2) / (2 * var))
        with torch.no_grad():
            sns.histplot(values.detach().numpy(), kde=True, stat='density')
        plt.plot(xs, ys, label='std Gaussian', c='r')
        plt.legend()
        plt.xlabel('values')
        plt.ylabel('values density')
        plt.title(title)
        plt.show()

    def test_pre_activations_init_dist(self):
        n_trials = 100
        std = self.ipllr_bias.std

        z = 2 * torch.rand(self.base_model_config.architecture["input_size"], requires_grad=False) - 1
        norm_2_z = torch.norm(z, 2).item()
        if np.abs(norm_2_z - 1) > 1e-6:
            z = z / norm_2_z  # normalized z

        hs = [[] for _ in range(self.ipllr_bias.n_layers)]
        for _ in range(n_trials):
            model = FcIPLLRBias(self.base_model_config, n_warmup_steps=1)
            model.eval()
            with torch.no_grad():
                x = z
                h = (model.width ** (-model.a[0])) * model.input_layer.forward(x)
                x = model.activation(h)
                hs[0].append(h)

                for l, layer in enumerate(model.intermediate_layers):
                    h = layer.forward((model.width ** (-model.a[l+1])) * x)
                    x = model.activation(h)
                    hs[l+1].append(h)

                h = model.output_layer.forward((model.width ** (-model.a[model.n_layers - 1])) * x)
                hs[model.n_layers - 1].append(h)

        hs = [torch.cat(pre_acts, dim=0) for pre_acts in hs]

        for l, h in enumerate(hs):
            if l == self.ipllr_bias.n_layers - 1:
                std = 1.0
            self.plot_parameter_values_histogram(h, std,
                                                 'Distribution of pre-activation values at layer {:,}'.format(l+1))

    def test_lrs(self):
        model = deepcopy(self.ipllr_bias)
        L = self.L
        m = self.width

        # initial model lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [1.0 for l in range(1, L+2)]
        self.assertSequenceEqual(model.lr_scales, expected_lr_scales)
        self.assertSequenceEqual(model.bias_lr_scales, expected_bias_lr_scales)

        # warm model lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [1.0 for l in range(1, L+2)]
        self.assertSequenceEqual(model.warm_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.warm_bias_lrs, expected_bias_warm_lr_scales)

        # initial scheduler lrs
        self.assertSequenceEqual(model.scheduler.initial_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_lr_scales)

        # warm scheduler lrs
        self.assertSequenceEqual(model.scheduler.warm_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.warm_bias_lrs, expected_bias_warm_lr_scales)

    def test_lrs_scale_bias_lr(self):
        config = deepcopy(self.base_model_config)
        config.architecture["scale_bias_lr"] = True
        model = FcIPLLRBias(config, n_warmup_steps=12)

        L = self.L
        m = self.width

        # initial model lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [m ** ((L - l + 2) / 2) for l in range(1, L+1)] + [1.0]
        self.assertSequenceEqual(model.lr_scales, expected_lr_scales)
        self.assertSequenceEqual(model.bias_lr_scales, expected_bias_lr_scales)

        # warm model lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [m for l in range(1, L+1)] + [1.0]
        self.assertSequenceEqual(model.warm_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.warm_bias_lrs, expected_bias_warm_lr_scales)

        # initial scheduler lrs
        self.assertSequenceEqual(model.scheduler.initial_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_lr_scales)

        # warm scheduler lrs
        self.assertSequenceEqual(model.scheduler.warm_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.warm_bias_lrs, expected_bias_warm_lr_scales)

    def test_lr_switch_scheduler(self):
        n = 10
        x = torch.rand((10, self.base_model_config.architecture["input_size"]), requires_grad=False)
        y = torch.randint(low=0, high=self.base_model_config.architecture["output_size"], size=(n,),
                          requires_grad=False)

        model = deepcopy(self.ipllr_bias)
        L = self.L
        m = self.width

        self.assertEqual(model.n_warmup_steps, 1)
        self.assertEqual(model.scheduler.n_warmup_steps, 1)

        # initial lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [1.0 for l in range(1, L+2)]

        # warm lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [1.0 for l in range(1, L+2)]

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_lr_scales[i - (self.L + 1)])

        # train for one step
        model.train()

        y_hat = model.forward(x)
        loss = model.loss(y_hat, y)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        model.eval()

        # check schedulers lrs (current, last) and param groups lrs
        self.assertSequenceEqual(model.scheduler._last_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler._last_bias_lrs, expected_bias_lr_scales)

        self.assertSequenceEqual(model.scheduler.current_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_warm_lr_scales)

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_warm_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_warm_lr_scales[i - (self.L + 1)])

    def test_lr_switch_scheduler_scale_bias(self):
        n = 10
        x = torch.rand((10, self.base_model_config.architecture["input_size"]), requires_grad=False)
        y = torch.randint(low=0, high=self.base_model_config.architecture["output_size"], size=(n,),
                          requires_grad=False)

        config = deepcopy(self.base_model_config)
        config.architecture["scale_bias_lr"] = True
        model = FcIPLLRBias(config, n_warmup_steps=12)

        L = self.L
        m = self.width

        self.assertEqual(model.n_warmup_steps, 1)
        self.assertEqual(model.scheduler.n_warmup_steps, 1)

        # initial lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [m ** ((L - l + 2) / 2) for l in range(1, L+1)] + [1.0]

        # warm lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [m for l in range(1, L+1)] + [1.0]

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_lr_scales[i - (self.L + 1)])

        # train for one step
        model.train()

        y_hat = model.forward(x)
        loss = model.loss(y_hat, y)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        model.eval()

        # check schedulers lrs (current, last) and param groups lrs
        self.assertSequenceEqual(model.scheduler._last_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler._last_bias_lrs, expected_bias_lr_scales)

        self.assertSequenceEqual(model.scheduler.current_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_warm_lr_scales)

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_warm_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_warm_lr_scales[i - (self.L + 1)])

    def test_calibrate_lr_switch_scheduler(self):
        n = 10
        x = torch.rand((10, self.base_model_config.architecture["input_size"]), requires_grad=False)
        y = torch.randint(low=0, high=self.base_model_config.architecture["output_size"], size=(n,),
                          requires_grad=False)

        config = deepcopy(self.base_model_config)
        config.scheduler.params = {'n_warmup_steps': self.n_warmup_steps,
                                   'calibrate_base_lr': True,
                                   'default_calibration': False}
        model = FcIPLLRBias(config, n_warmup_steps=12, lr_calibration_batches=[(x, y), (x, y)])

        L = self.L
        m = self.width

        self.assertEqual(model.n_warmup_steps, 1)
        self.assertEqual(model.scheduler.n_warmup_steps, 1)

        # initial lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [1.0 for l in range(1, L+2)]

        # warm lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [1.0 for l in range(1, L+2)]

        # train for one step
        model.train()

        y_hat = model.forward(x)
        loss = model.loss(y_hat, y)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        model.eval()

        # check schedulers lrs (current, last) and param groups lrs
        self.assertSequenceEqual(model.scheduler._last_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler._last_bias_lrs, expected_bias_lr_scales)

        self.assertSequenceEqual(model.scheduler.current_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_warm_lr_scales)

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_warm_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_warm_lr_scales[i - (self.L + 1)])

    def test_calibrate_lr_default_switch_scheduler(self):
        n = 10
        x = torch.rand((10, self.base_model_config.architecture["input_size"]), requires_grad=False)
        y = torch.randint(low=0, high=self.base_model_config.architecture["output_size"], size=(n,),
                          requires_grad=False)

        config = deepcopy(self.base_model_config)
        config.scheduler.params = {'n_warmup_steps': self.n_warmup_steps,
                                   'calibrate_base_lr': True,
                                   'default_calibration': True}
        model = FcIPLLRBias(config, n_warmup_steps=12, lr_calibration_batches=[(x, y), (x, y)])

        self.assertEqual(model.scheduler.initial_base_lrs, 1.0)

        L = self.L
        m = self.width

        self.assertEqual(model.n_warmup_steps, 1)
        self.assertEqual(model.scheduler.n_warmup_steps, 1)

        # initial lrs
        expected_lr_scales = [m ** ((L + 1) / 2)] + [m ** ((L - l + 4) / 2) for l in range(2, L+1)] + [m]
        expected_bias_lr_scales = [1.0 for l in range(1, L+2)]

        # warm lrs
        expected_warm_lr_scales = [m] + [m ** 2 for l in range(2, L+1)] + [m]
        expected_bias_warm_lr_scales = [1.0 for l in range(1, L+2)]

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], expected_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_lr_scales[i - (self.L + 1)])

        # train for one step
        model.train()

        y_hat = model.forward(x)
        loss = model.loss(y_hat, y)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        model.eval()

        # check schedulers lrs (current, last) and param groups lrs
        self.assertSequenceEqual(model.scheduler._last_lrs, expected_lr_scales)
        self.assertSequenceEqual(model.scheduler._last_bias_lrs, expected_bias_lr_scales)

        self.assertSequenceEqual(model.scheduler.current_lrs, expected_warm_lr_scales)
        self.assertSequenceEqual(model.scheduler.current_bias_lrs, expected_bias_warm_lr_scales)

        # check param groups initial lrs
        for i, param in enumerate(model.optimizer.param_groups):
            if 'weight' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_warm_lr_scales[i])
            elif 'bias' in param['name']:
                self.assertEqual(param['lr'], model.base_lr * expected_bias_warm_lr_scales[i - (self.L + 1)])


if __name__ == '__main__':
    unittest.main()
