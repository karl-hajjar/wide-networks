import unittest
import os
import os
from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F

from utils.tools import read_yaml
from pytorch.configs.base import BaseConfig
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
        Ls = [4]
        n_batches = 10
        base_lrs = [1.0]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)

        for base_lr in base_lrs:
            config.optimizer.params['lr'] = base_lr
            for L in Ls:
                config.architecture['n_layers'] = L + 1
                for width in widths:
                    config.architecture['width'] = width

                    config.scheduler.params['calibrate_base_lr'] = False
                    ipllr = FcIPLLR(config, n_warmup_steps=12)

                    config.scheduler.params['calibrate_base_lr'] = True
                    ipllr_calib = FcIPLLR(config, n_warmup_steps=12, lr_calibration_batches=batches)

                    ipllr_calib.copy_initial_params_from_model(ipllr)
                    ipllr_calib.initialize_params()

                    ipllr_init = deepcopy(ipllr)
                    ipllr_calib_init = deepcopy(ipllr_calib)

                    # first train the models for one step
                    x, y = batches[0]
                    self._train_models_one_step(ipllr, ipllr_calib, x, y)

                    reduced_batches = batches[1: n_batches + 1]
                    ipllr_contribs = self._compute_contributions('ipllr', ipllr, ipllr_init, reduced_batches)
                    ipllr_calib_contribs = self._compute_contributions('ipllr_calib', ipllr_calib, ipllr_calib_init,
                                                                       reduced_batches)

                    print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                    print('ipllr contributions per layer : ')
                    print(ipllr_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                    print('')

                    print('calibrated ipllr contributions per layer : ')
                    print(ipllr_calib_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                    print('\n\n')

    def _train_models_one_step(self, model1, model2, x, y):
        model1.train()
        model2.train()

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
        print('loss derivatives for model 1 :', self.batch_size * y_hat_1.grad)
        print('loss derivatives for model 2 :', self.batch_size * y_hat_2.grad)
        print('average training loss for model1 :', np.mean(loss_1.item()))
        print('average training loss for model2 :', np.mean(loss_2.item()))
        print('')

        # first weight update
        model1.optimizer.step()
        model2.optimizer.step()

        if hasattr(model1, "scheduler") and model1.scheduler is not None:
            model1.scheduler.step()
        model2.scheduler.step()

    @staticmethod
    def _compute_contributions(model_name, model, model_init, batches):
        model.eval()
        model_init.eval()

        contributions_df = pd.DataFrame(columns=['model', 'layer', 'init', 'update', 'total', 'id'])
        contributions_df.loc[:, ['init', 'update', 'total', 'id']] = \
            contributions_df.loc[:, ['init', 'update', 'total', 'id']].astype(float)

        idx = 0

        L = model.n_layers - 1
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(batches):
                x, y = batch
                with torch.no_grad():
                    x_0 = x.clone().detach()

                # input layer
                Delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                            model_init.input_layer.weight.data)
                Delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                            model_init.input_layer.bias.data)

                init_contrib = model_init.layer_scales[0] * model_init.input_layer.forward(x)
                update_contrib = F.linear(x, Delta_W, Delta_b)
                total_contrib = model.layer_scales[0] * model.input_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                              rtol=1e-3, atol=1e-3)

                # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                #     [model_name, 1, init_contrib.mean().item(), update_contrib.mean().item(),
                #      total_contrib.mean().item(), i]
                contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    [model_name, 1, init_contrib.abs().mean().item(), update_contrib.abs().mean().item(),
                     total_contrib.abs().mean().item(), i]
                idx += 1

                x = model.activation(total_contrib)

                # intermediate layer grads
                for l in range(2, L + 1):
                    layer_key = "layer_{:,}_intermediate".format(l)
                    layer = getattr(model.intermediate_layers, layer_key)
                    init_layer = getattr(model_init.intermediate_layers, layer_key)

                    Delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)

                    init_contrib = model_init.layer_scales[l - 1] * init_layer.forward(x)
                    update_contrib = F.linear(x, Delta_W)
                    total_contrib = model.layer_scales[l - 1] * layer.forward(x)

                    torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                                  rtol=1e-2, atol=1e-2)

                    # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    #     [model_name, l, init_contrib.mean().item(), update_contrib.mean().item(),
                    #      total_contrib.mean().item(), i]
                    contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                        [model_name, l, init_contrib.abs().mean().item(), update_contrib.abs().mean().item(),
                         total_contrib.abs().mean().item(), i]
                    idx += 1

                    x = model.activation(total_contrib)

                # output layer
                Delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                            model_init.output_layer.weight.data)
                init_contrib = model_init.layer_scales[L] * model_init.output_layer.forward(x)
                update_contrib = F.linear(x, Delta_W)
                total_contrib = model.layer_scales[L] * model.output_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                              rtol=1e-2, atol=1e-2)

                # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                #     [model_name, L + 1, init_contrib.mean().item(), update_contrib.mean().item(),
                #      total_contrib.mean().item(), i]
                contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    [model_name, L + 1, init_contrib.abs().mean().item(), update_contrib.abs().mean().item(),
                     total_contrib.abs().mean().item(), i]
                idx += 1

                y_hat_debug = total_contrib
                y_hat = model.forward(x_0)

                torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

                losses.append(model.loss(y_hat, y).item())

        print('average validation loss for {} : {}'.format(model_name, np.mean(losses)))
        return contributions_df

    def test_scales_multiple_steps(self):
        n_steps = 200
        widths = [1024]
        Ls = [6]
        n_batches = 10
        base_lrs = [0.1]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)
        print('len(batches) :', len(batches))

        for base_lr in base_lrs:
            config.optimizer.params['lr'] = base_lr
            for L in Ls:
                config.architecture['n_layers'] = L + 1
                for width in widths:
                    config.architecture['width'] = width

                    config.scheduler.params['calibrate_base_lr'] = True
                    config.scheduler.params['default_calibration'] = True
                    ipllr = FcIPLLR(config, n_warmup_steps=12)

                    config.scheduler.params['calibrate_base_lr'] = True
                    config.scheduler.params['default_calibration'] = False
                    ipllr_calib = FcIPLLR(config, n_warmup_steps=12, lr_calibration_batches=batches)

                    ipllr_calib.copy_initial_params_from_model(ipllr)
                    ipllr_calib.initialize_params()

                    ipllr_init = deepcopy(ipllr)
                    ipllr_calib_init = deepcopy(ipllr_calib)

                    for step in range(n_steps):
                        print('##### step {} ####'.format(step))
                        # first train the models for one step
                        x, y = batches[step]
                        self._train_models_one_step(ipllr, ipllr_calib, x, y)

                        batch_nb = 1 + step * n_batches % len(batches)
                        print('batch_nb:', batch_nb)
                        reduced_batches = batches[batch_nb: batch_nb + n_batches]
                        print('len(reduced_batches) at step :', len(reduced_batches))
                        ipllr_contribs = self._compute_contributions('ipllr', ipllr, ipllr_init, reduced_batches)
                        ipllr_calib_contribs = self._compute_contributions('ipllr_calib', ipllr_calib, ipllr_calib_init,
                                                                           reduced_batches)
                        print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                        print('ipllr contributions per layer : ')
                        print(ipllr_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('')

                        print('calibrated ipllr contributions per layer : ')
                        print(ipllr_calib_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('\n\n')

    def test_scales_multiple_steps_muP(self):
        n_steps = 10
        widths = [1024]
        Ls = [6]
        n_batches = 10
        base_lrs = [0.01, 0.1]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)
        print('len(batches) :', len(batches))
        for L in Ls:
            config.architecture['n_layers'] = L + 1
            for width in widths:
                config.architecture['width'] = width

                config.scheduler = None
                base_muP = FCmuP(config)

                for base_lr in base_lrs:
                    config.optimizer.params['lr'] = base_lr

                    config.scheduler = None
                    muP = FCmuP(config)

                    scheduler_config = {'calibrate_base_lr': True, 'default_calibration': False}
                    config.scheduler = BaseConfig(scheduler_config)
                    # config.scheduler.params['calibrate_base_lr'] = True
                    # config.scheduler.params['default_calibration'] = False
                    ipllr_calib = FcIPLLR(config, n_warmup_steps=12, lr_calibration_batches=batches)

                    # set init from same model
                    muP.copy_initial_params_from_model(base_muP)
                    muP.initialize_params()

                    ipllr_calib.copy_initial_params_from_model(base_muP)
                    ipllr_calib.initialize_params()

                    muP_init = deepcopy(muP)
                    ipllr_calib_init = deepcopy(ipllr_calib)

                    for step in range(n_steps):
                        print('##### step {} ####'.format(step))
                        # first train the models for one step
                        x, y = batches[step]
                        self._train_models_one_step(muP, ipllr_calib, x, y)

                        # batch_nb = 1 + step * n_batches % len(batches)
                        # print('batch_nb:', batch_nb)
                        # reduced_batches = batches[batch_nb: batch_nb + n_batches]
                        reduced_batches = batches[-n_batches:]
                        # print('len(reduced_batches) at step {} : {}'.format(step, len(reduced_batches)))
                        muP_contribs = self._compute_contributions('muP', muP, muP_init, reduced_batches)
                        ipllr_calib_contribs = self._compute_contributions('ipllr_calib', ipllr_calib, ipllr_calib_init,
                                                                           reduced_batches)
                        print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                        print('muP contributions per layer : ')
                        print(muP_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('')

                        print('calibrated ipllr contributions per layer : ')
                        print(ipllr_calib_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('\n\n')

    def test_scales_vs_previous_multiple_steps_muP(self):
        n_steps = 10
        widths = [1024]
        Ls = [6]
        n_batches = 10
        base_lrs = [0.1, 0.01, 0.001]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)
        print('len(batches) :', len(batches))

        for L in Ls:
            config.architecture['n_layers'] = L + 1
            for width in widths:
                config.architecture['width'] = width

                config.scheduler = None
                base_muP = FCmuP(config)

                for base_lr in base_lrs:
                    config.optimizer.params['lr'] = base_lr

                    config.scheduler = None
                    muP = FCmuP(config)

                    scheduler_config = {'calibrate_base_lr': True, 'default_calibration': False}
                    config.scheduler = BaseConfig(scheduler_config)
                    # config.scheduler.params['calibrate_base_lr'] = True
                    # config.scheduler.params['default_calibration'] = False
                    ipllr_calib = FcIPLLR(config, n_warmup_steps=12, lr_calibration_batches=batches)

                    # set init from same model
                    muP.copy_initial_params_from_model(base_muP)
                    muP.initialize_params()

                    ipllr_calib.copy_initial_params_from_model(muP)
                    ipllr_calib.initialize_params()

                    for step in range(n_steps):
                        print('##### step {} ####'.format(step))
                        # first train the models for one step
                        x, y = batches[step]

                        # copy models at current step
                        muP_previous = deepcopy(muP)
                        ipllr_calib_previous = deepcopy(ipllr_calib)

                        # train for oone step
                        self._train_models_one_step(muP, ipllr_calib, x, y)

                        # batch_nb = 1 + step * n_batches % len(batches)
                        # print('batch_nb:', batch_nb)
                        reduced_batches = batches[-n_batches:]
                        # print('len(reduced_batches) at step {} : {}'.format(step, len(reduced_batches)))
                        muP_contribs = self._compute_contributions('muP', muP, muP_previous, reduced_batches)
                        ipllr_calib_contribs = self._compute_contributions('ipllr_calib', ipllr_calib,
                                                                           ipllr_calib_previous, reduced_batches)
                        print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                        print('muP contributions per layer : ')
                        print(muP_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('')

                        print('calibrated ipllr contributions per layer : ')
                        print(ipllr_calib_contribs.groupby(by='layer')[['init', 'update', 'total']].mean())
                        print('\n\n')

    def test_scales_with_previous_multiple_steps_muP(self):
        n_steps = 80
        widths = [1024]
        Ls = [6]
        n_batches = 10
        base_lrs = [0.001]
        config = deepcopy(self.base_model_config)

        batches = list(self.train_data_loader)
        print('len(batches) :', len(batches))

        for L in Ls:
            config.architecture['n_layers'] = L + 1
            for width in widths:
                config.architecture['width'] = width

                config.scheduler = None
                base_muP = FCmuP(config)

                for base_lr in base_lrs:
                    config.optimizer.params['lr'] = base_lr

                    config.scheduler = None
                    muP = FCmuP(config)

                    scheduler_config = {'calibrate_base_lr': True, 'default_calibration': False}
                    config.scheduler = BaseConfig(scheduler_config)
                    # config.scheduler.params['calibrate_base_lr'] = True
                    # config.scheduler.params['default_calibration'] = False
                    ipllr_calib = FcIPLLR(config, n_warmup_steps=12, lr_calibration_batches=batches)

                    # set init from same model
                    muP.copy_initial_params_from_model(base_muP)
                    muP.initialize_params()

                    ipllr_calib.copy_initial_params_from_model(muP)
                    ipllr_calib.initialize_params()

                    muP_init = deepcopy(muP)
                    ipllr_calib_init = deepcopy(ipllr_calib)

                    for step in range(n_steps):
                        print('##### step {} ####'.format(step))
                        # first train the models for one step
                        x, y = batches[step]

                        # copy models at current step
                        muP_previous = deepcopy(muP)
                        ipllr_calib_previous = deepcopy(ipllr_calib)

                        # train for oone step
                        self._train_models_one_step(muP, ipllr_calib, x, y)

                        # batch_nb = 1 + step * n_batches % len(batches)
                        # print('batch_nb:', batch_nb)
                        next_batch = batches[step % len(batches)]
                        # print('len(reduced_batches) at step {} : {}'.format(step, len(reduced_batches)))
                        muP_contribs = \
                            self._compute_contributions_with_previous('muP', muP, muP_init, muP_previous, [next_batch])
                        ipllr_calib_contribs = \
                            self._compute_contributions_with_previous('ipllr_calib', ipllr_calib, ipllr_calib_init,
                                                                      ipllr_calib_previous, [next_batch])
                        print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                        print('muP contributions per layer : ')
                        print(muP_contribs.groupby(by='layer')[['init', 'previous_h', 'previous_Delta_h', 'delta_h', 'Delta_h',
                                                                'total']].mean())
                        print('')

                        print('calibrated ipllr contributions per layer : ')
                        print(ipllr_calib_contribs.groupby(by='layer')[['init', 'previous_h', 'previous_Delta_h',
                                                                        'delta_h', 'Delta_h',  'total']].mean())
                        print('\n\n')

    @staticmethod
    def _compute_contributions_with_previous(model_name, model, model_init, model_previous, batches):
        model.eval()
        model_init.eval()
        model_previous.eval()

        contributions_df = pd.DataFrame(columns=['model', 'layer', 'init', 'previous_h', 'previous_Delta_h', 'Delta_h',
                                                 'delta_h', 'total', 'id'])
        contributions_df.loc[:, ['init', 'previous_h', 'previous_Delta_h', 'Delta_h', 'delta_h', 'total', 'id']] = \
            contributions_df.loc[:, ['init', 'previous_h', 'previous_Delta_h', 'Delta_h', 'delta_h', 'total', 'id']].\
                astype(float)

        idx = 0

        L = model.n_layers - 1
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(batches):
                x, y = batch
                with torch.no_grad():
                    x_0 = x.clone().detach()

                # input layer
                Delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                            model_init.input_layer.weight.data)
                Delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                            model_init.input_layer.bias.data)

                previous_Delta_W = (model.width ** (-model.a[0])) * (model_previous.input_layer.weight.data -
                                                                     model_init.input_layer.weight.data)
                previous_Delta_b = (model.width ** (-model.a[0])) * (model_previous.input_layer.bias.data -
                                                                     model_init.input_layer.bias.data)

                delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                            model_previous.input_layer.weight.data)
                delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                            model_previous.input_layer.bias.data)

                init_contrib = model_init.layer_scales[0] * model_init.input_layer.forward(x)
                previous_h = model.layer_scales[0] * model_previous.input_layer.forward(x)

                previous_Delta_h = F.linear(x, previous_Delta_W, previous_Delta_b)
                Delta_h = F.linear(x, Delta_W, Delta_b)
                delta_h = F.linear(x, delta_W, delta_b)
                total_contrib = model.layer_scales[0] * model.input_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                              rtol=1e-3, atol=1e-3)

                # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                #     [model_name, 1, init_contrib.mean().item(), update_contrib.mean().item(),
                #      total_contrib.mean().item(), i]
                contributions_df.loc[idx, ['model', 'layer', 'init', 'previous_h', 'previous_Delta_h', 'Delta_h',
                                           'delta_h', 'total', 'id']] = \
                    [model_name, 1, init_contrib.abs().mean().item(), previous_h.abs().mean().item(),
                     previous_Delta_h.abs().mean().item(), Delta_h.abs().mean().item(), delta_h.abs().mean().item(),
                     total_contrib.abs().mean().item(), i]
                idx += 1

                x = model.activation(total_contrib)

                # intermediate layer grads
                for l in range(2, L + 1):
                    layer_key = "layer_{:,}_intermediate".format(l)
                    layer = getattr(model.intermediate_layers, layer_key)
                    init_layer = getattr(model_init.intermediate_layers, layer_key)
                    previous_layer = getattr(model_previous.intermediate_layers, layer_key)

                    Delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)
                    previous_Delta_W = (model.width ** (-model.a[l - 1])) * (previous_layer.weight.data -
                                                                             init_layer.weight.data)
                    delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - previous_layer.weight.data)

                    init_contrib = model_init.layer_scales[l - 1] * init_layer.forward(x)
                    previous_h = model.layer_scales[l - 1] * previous_layer.forward(x)

                    previous_Delta_h = F.linear(x, previous_Delta_W)
                    Delta_h = F.linear(x, Delta_W)
                    delta_h = F.linear(x, delta_W)
                    total_contrib = model.layer_scales[l - 1] * layer.forward(x)

                    torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                                  rtol=1e-2, atol=1e-2)

                    # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    #     [model_name, l, init_contrib.mean().item(), update_contrib.mean().item(),
                    #      total_contrib.mean().item(), i]
                    contributions_df.loc[idx, ['model', 'layer', 'init', 'previous_h',  'previous_Delta_h',  'Delta_h',
                                               'delta_h', 'total', 'id']] = \
                        [model_name, l, init_contrib.abs().mean().item(), previous_h.abs().mean().item(),
                         previous_Delta_h.abs().mean().item(), Delta_h.abs().mean().item(), delta_h.abs().mean().item(),
                         total_contrib.abs().mean().item(), i]
                    idx += 1

                    x = model.activation(total_contrib)

                # output layer
                Delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                            model_init.output_layer.weight.data)
                previous_Delta_W = (model.width ** (-model.a[L])) * (model_previous.output_layer.weight.data -
                                                                     model_init.output_layer.weight.data)
                delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                            model_previous.output_layer.weight.data)

                init_contrib = model_init.layer_scales[L] * model_init.output_layer.forward(x)
                previous_h = model.layer_scales[L] * model_previous.output_layer.forward(x)

                previous_Delta_h = F.linear(x, previous_Delta_W)
                Delta_h = F.linear(x, Delta_W)
                delta_h = F.linear(x, delta_W)
                total_contrib = model.layer_scales[L] * model.output_layer.forward(x)

                torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                              rtol=1e-2, atol=1e-2)

                # contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                #     [model_name, L + 1, init_contrib.mean().item(), update_contrib.mean().item(),
                #      total_contrib.mean().item(), i]
                contributions_df.loc[idx, ['model', 'layer', 'init', 'previous_h',  'previous_Delta_h',  'Delta_h',
                                           'delta_h', 'total', 'id']] = \
                    [model_name, L + 1, init_contrib.abs().mean().item(), previous_h.abs().mean().item(),
                     previous_Delta_h.abs().mean().item(), Delta_h.abs().mean().item(), delta_h.abs().mean().item(),
                     total_contrib.abs().mean().item(), i]
                idx += 1

                y_hat_debug = total_contrib
                y_hat = model.forward(x_0)

                torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

                losses.append(model.loss(y_hat, y).item())

        print('average validation loss for {} : {}'.format(model_name, np.mean(losses)))
        return contributions_df


if __name__ == '__main__':
    unittest.main()
