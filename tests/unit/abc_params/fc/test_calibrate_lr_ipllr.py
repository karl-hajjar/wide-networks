import unittest
import os
import os
from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
from pytorch.models.abc_params.fully_connected.muP import FCmuP
from pytorch.models.abc_params.fully_connected.ntk import FCNTK
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.data.mnist import load_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')

ROOT = os.path.dirname(TESTS_DIR)
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', 'fc_ipllr_mnist.yaml')


class TestCalibrateBaseLR(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)

        self.input_size = config_dict['architecture']['input_size']
        self.base_lr = config_dict['optimizer']['params']['lr']
        self.n_warmup_steps = 1
        self.batch_size = 128

        self.width = config_dict['architecture']['width']
        self.L = 3
        config_dict['architecture']['n_layers'] = self.L + 1
        config_dict['optimizer']['params']['lr'] = self.base_lr
        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': self.n_warmup_steps,
                                               'calibrate_base_lr': True}}

        self.base_model_config = ModelConfig(config_dict)
        training_dataset, _ = load_data(download=False, flatten=True)
        self.train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=self.batch_size)
        self.batches = list(self.train_data_loader)

    def test_basic_calibration(self):
        ipllr = FcIPLLR(self.base_model_config, n_warmup_steps=12, lr_calibration_batches=self.batches)
        self.assertEqual(True, True)

    def test_scales_with_and_without_calibration(self):
        widths = [1024]
        Ls = [43]
        n_batches = 10
        base_lrs = [0.1]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)

        for base_lr in base_lrs:
            config.optimizer.params['lr'] = base_lr
            for L in Ls:
                config.architecture['n_layers'] = L
                for width in widths:
                    config.architecture['width'] = width

                    config.scheduler.params['calibrate_base_lr'] = False
                    ipllr = FcIPLLR(self.base_model_config, n_warmup_steps=12)

                    config.scheduler.params['calibrate_base_lr'] = True
                    ipllr_calib = FcIPLLR(self.base_model_config, n_warmup_steps=12, lr_calibration_batches=self.batches)

                    ipllr_calib.copy_initial_params_from_model(ipllr)
                    ipllr_calib.initialize_params()

                    ipllr_init = deepcopy(ipllr)
                    ipllr_calib_init = deepcopy(ipllr_calib)

                    ipllr.train()
                    ipllr_calib.train()

                    # first train the models for one step
                    x, y = batches[0]
                    self._train_models_one_step(ipllr, ipllr_calib, x, y)

                    ipllr.eval()
                    ipllr_calib.eval()

                    ipllr_contribs = self._compute_contributions('ipllr', ipllr, ipllr_init, batches[1:n_batches + 1])
                    ipllr_calib_contribs = self._compute_contributions('ipllr_calib', ipllr_calib, ipllr_calib,
                                                                       batches[1:n_batches + 1])

                    print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                    print('ipllr contributions per layer : ')
                    print(ipllr_contributions.groupby(by='layer')[['init', 'update', 'total']].mean())
                    print('')

                    print('muP contributions per layer : ')
                    print(muP_contributions.groupby(by='layer')[['init', 'update', 'total']].mean())
                    print('\n\n')


    @staticmethod
    def _train_models_one_step(model1, model2, x, y):
        # set gradients to 0
        model1.optimizer.zero_grad()
        model2.optimizer.zero_grad()

        # outputs at initialization
        y_hat_1 = model1.forward(x)
        y_hat_1.retain_grad()
        loss_1 = model1.loss(y_hat_1, y)

        y_hat_2 = model2.forward(x)
        y_hat_2.retain_grad()
        loss_2 = model2.loss(y_hat_2, y)

        # gradients at initialization
        loss_1.backward()
        loss_2.backward()

        print('input abs mean in training: ', x.abs().mean().item())
        print('loss derivatives for model 1 :', y_hat_1.grad)
        print('loss derivatives for model 2 :', y_hat_2.grad)
        print('')

        # first weight update
        model1.optimizer.step()
        model2.optimizer.step()

    @staticmethod
    def _compute_contributions(model_name, model, model_init, batches):
        contributions_df = pd.DataFrame(columns=['model', 'layer', 'init', 'update', 'total', 'id'])
        contributions_df.loc[:, ['init', 'update', 'total', 'id']] = \
            contributions_df.loc[:, ['init', 'update', 'total', 'id']].astype(float)

        idx = 0

        L = model.n_layers - 1
        with torch.no_grad():

            for i, batch in enumerate(batches):
                x, y = batch
                with torch.no_grad():
                    x_0 = x.clone().detach()

                # input layer
                Delta_w = model.input_layer.weight.data - model_init.input_layer.weight.data
                init_contrib = model_init.layer_scales[0] * model_init.input_layer.forward(x)
                update_contrib = model.layer_scales[0] * x.matmul(Delta_w.t())
                total_contrib = model.layer_scales[0] * model.input_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                              rtol=1e-3, atol=1e-3)

                contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    [model_name, 1, init_contrib.mean().item(), update_contrib.mean().item(),
                     total_contrib.mean().item(), i]
                idx += 1

                x = model.activation(total_contrib)

                # intermediate layer grads
                for l in range(2, L + 1):
                    layer_key = "layer_{:,}_intermediate".format(l)
                    layer = getattr(model.intermediate_layers, layer_key)
                    init_layer = getattr(model_init.intermediate_layers, layer_key)

                    Delta_w = layer.weight.data - init_layer.weight.data

                    init_contrib = model_init.layer_scales[l - 1] * init_layer.forward(x)
                    update_contrib = model.layer_scales[l - 1] * x.matmul(Delta_w.t())
                    total_contrib = model.layer_scales[l - 1] * layer.forward(x)

                    torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                                  rtol=1e-3, atol=1e-3)

                    contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                        [model_name, l, init_contrib.mean().item(), update_contrib.mean().item(),
                         total_contrib.mean().item(), i]
                    idx += 1

                    x = model.activation(total_contrib)

                # output layer
                Delta_w = model.output_layer.weight.data - model_init.output_layer.weight.data
                init_contrib = model_init.layer_scales[L] * model_init.output_layer.forward(x)
                update_contrib = model.layer_scales[L] * x.matmul(Delta_w.t())
                total_contrib = model.layer_scales[L] * model.output_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                              rtol=1e-3, atol=1e-3)

                contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    [model_name, L + 1, init_contrib.mean().item(), update_contrib.mean().item(),
                     total_contrib.mean().item(), i]
                idx += 1

                y_hat_debug = total_contrib
                y_hat = model.forward(x_0)

                torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

        return contributions_df


if __name__ == '__main__':
    unittest.main()
