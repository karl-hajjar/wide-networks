import unittest
import os
from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
from pytorch.models.abc_params.fully_connected.muP import FCmuP
from pytorch.models.abc_params.fully_connected.ntk import FCNTK
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FIGURES_DIR = os.path.join(TESTS_DIR, 'unit/outputs/figures/abc_params')

CONFIG_PATH = os.path.join(TESTS_DIR, 'resources', 'fc_abc_config.yaml')


class TestFcIPLLR(unittest.TestCase):
    def setUp(self) -> None:
        config_dict = read_yaml(CONFIG_PATH)

        self.input_size = config_dict['architecture']['input_size']
        self.base_lr = config_dict['optimizer']['params']['lr']
        self.n_warmup_steps = 2
        self.width = config_dict['architecture']['width']
        self.L = config_dict['architecture']['n_layers'] - 1
        config_dict['optimizer']['params']['lr'] = self.base_lr
        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': self.n_warmup_steps}}

        self.base_model_config = ModelConfig(config_dict)
        self.ipllr = FcIPLLR(self.base_model_config, n_warmup_steps=4)

    def test_scheduler_initial_state(self):
        Ls = [2, 3, 4, 5, 10]
        for L in Ls:
            model_config = deepcopy(self.base_model_config)
            model_config.architecture['n_layers'] = L + 1
            ipllr = FcIPLLR(model_config, self.width)

            scheduler = ipllr.scheduler

            self.assertEqual(ipllr.width, self.width)
            self.assertEqual(scheduler._step_count, 1)
            self.assertEqual(scheduler.base_lr, self.base_lr)
            self.assertEqual(ipllr.base_lr, self.base_lr)
            self.assertEqual(scheduler.n_warmup_steps, self.n_warmup_steps)

            self.assertEqual(len(scheduler.initial_lrs), ipllr.n_layers)
            self.assertEqual(len(scheduler.warm_lrs), ipllr.n_layers)

            self.assertSequenceEqual(scheduler.initial_lrs,
                                     [self.width ** (+ (L + 1) / 2)] +
                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                     [self.width ** (+ (L + 1) / 2)])
            self.assertSequenceEqual(scheduler.warm_lrs,
                                     [self.width ** (+ 1)] +
                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                     [self.width ** (+ 1)])
            self.assertSequenceEqual(scheduler.current_lrs,
                                     [self.width ** (+ (L + 1) / 2)] +
                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                     [self.width ** (+ (L + 1) / 2)])
            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                       [self.base_lr * self.width ** (+ (L + 1) / 2)] +
                                       [self.base_lr * self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                       [self.base_lr * self.width ** (+ (L + 1) / 2)])
            self.test_param_groups_lrs(scheduler.optimizer.param_groups,
                                       [self.base_lr * self.width ** (+ (L + 1) / 2)] +
                                       [self.base_lr * self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                       [self.base_lr * self.width ** (+ (L + 1) / 2)])

    def test_param_groups_lrs(self, param_groups, expected_lrs):
        self.assertSequenceEqual([param['lr'] for param in param_groups], expected_lrs)

    def test_reference_scheduler_optimizer(self):
        modified_lrs = [l + 0.5 for l in range(self.ipllr.n_layers)]
        self.assertTrue(self.ipllr.scheduler.optimizer == self.ipllr.optimizer)

        # modify scheduler.optimizer lrs
        for i, param_group in enumerate(self.ipllr.scheduler.optimizer.param_groups):
            param_group['lr'] = modified_lrs[i]

        # check that modification happened in ipllr.optimizer
        self.test_param_groups_lrs(self.ipllr.optimizer.param_groups, modified_lrs)

    def test_switch(self):
        warmup_steps = [1, 4, 5, 10, 20]
        Ls = [2, 3, 4, 5, 10, 12]
        base_lrs = [0.001, 0.01, 0.02, 0.5, 1.0]
        
        n_samples = 10000
        batch_size = 64
        n_batches = (n_samples // batch_size) + 1

        xs = torch.rand(size=(n_samples, self.input_size))
        ys = torch.randint(high=10, size=(n_samples,))
        batches = [(xs[i * batch_size: (i+1) * batch_size, ], ys[i * batch_size: (i+1) * batch_size])
                   for i in range(n_batches)]

        for n_warmup_steps in warmup_steps:
            for L in Ls:
                for base_lr in base_lrs:
                    model_config = deepcopy(self.base_model_config)
                    model_config.scheduler.params['n_warmup_steps'] = n_warmup_steps
                    model_config.architecture['n_layers'] = L + 1
                    model_config.optimizer.params['lr'] = base_lr
                    ipllr = FcIPLLR(model_config)
                    ipllr.train()
        
                    self.assertEqual(ipllr.width, self.width)
        
                    for i, batch in enumerate(batches):
                        x, y = batch
                        y_hat = ipllr.forward(x)
                        loss = ipllr.loss(y_hat, y)
        
                        loss.backward()
                        ipllr.optimizer.step()
        
                        self.assertEqual(ipllr.scheduler._step_count, i + 1)
                        self.assertEqual(ipllr.base_lr, base_lr)
                        self.assertEqual(ipllr.scheduler.base_lr, base_lr)
                        self.assertEqual(ipllr.scheduler.optimizer, ipllr.optimizer)
                        if i < n_warmup_steps:
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ (L + 1) / 2)] +
                                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                                     [self.width ** (+ (L + 1) / 2)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ (L + 1) / 2)] +
                                                       [base_lr * self.width ** (+ (L + 2) / 2) for _ in
                                                        range(1, L)] +
                                                       [base_lr * self.width ** (+ (L + 1) / 2)])
                        elif i == n_warmup_steps:  # switch in lrs must have happened
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ 1)] +
                                                       [base_lr * self.width ** (+ 2) for _ in range(1, L)] +
                                                       [base_lr * self.width ** (+ 1)])
                            self.assertSequenceEqual(ipllr.scheduler._last_lrs,
                                                     [self.width ** (+ (L + 1) / 2)] +
                                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                                     [self.width ** (+ (L + 1) / 2)])
                        else:
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ 1)] +
                                                       [base_lr * self.width ** (+ 2) for _ in range(1, L)] +
                                                       [base_lr * self.width ** (+ 1)])
                            self.assertSequenceEqual(ipllr.scheduler._last_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])
        
                        ipllr.scheduler.step()
        
                        self.assertEqual(ipllr.scheduler._step_count, i + 2)
                        if i < n_warmup_steps - 1:
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ (L + 1) / 2)] +
                                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                                     [self.width ** (+ (L + 1) / 2)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ (L + 1) / 2)] +
                                                       [base_lr * self.width ** (+ (L + 2) / 2) for _ in
                                                        range(1, L)] +
                                                       [base_lr * self.width ** (+ (L + 1) / 2)])
                        elif i == n_warmup_steps - 1:  # switch in lrs must have happened
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ 1)] +
                                                       [base_lr * self.width ** (+ 2) for _ in range(1, L)] +
                                                       [base_lr * self.width ** (+ 1)])
                            self.assertSequenceEqual(ipllr.scheduler._last_lrs,
                                                     [self.width ** (+ (L + 1) / 2)] +
                                                     [self.width ** (+ (L + 2) / 2) for _ in range(1, L)] +
                                                     [self.width ** (+ (L + 1) / 2)])
                        else:
                            self.assertSequenceEqual(ipllr.scheduler.current_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])
                            self.test_param_groups_lrs(ipllr.optimizer.param_groups,
                                                       [base_lr * self.width ** (+ 1)] +
                                                       [base_lr * self.width ** (+ 2) for _ in range(1, L)] +
                                                       [base_lr * self.width ** (+ 1)])
                            self.assertSequenceEqual(ipllr.scheduler._last_lrs,
                                                     [self.width ** (+ 1)] +
                                                     [self.width ** (+ 2) for _ in range(1, L)] +
                                                     [self.width ** (+ 1)])

    def test_ipllr_vs_ip(self):
        n_warmup_steps = int(10e5)
        n_samples = 10000
        batch_size = 128
        n_batches = (n_samples // batch_size) + 1

        xs = torch.rand(size=(n_samples, self.input_size))
        ys = torch.randint(high=10, size=(n_samples,))
        batches = [(xs[i * batch_size: (i+1) * batch_size, ], ys[i * batch_size: (i+1) * batch_size])
                   for i in range(n_batches)]

        Ls = [2, 3, 4, 5, 10]
        base_lrs = [0.001, 0.01, 0.1]

        for L in Ls:
            for base_lr in base_lrs:
                model_config = deepcopy(self.base_model_config)
                model_config.scheduler.params['n_warmup_steps'] = n_warmup_steps
                model_config.architecture['n_layers'] = L + 1
                model_config.optimizer.params['lr'] = base_lr

                ipllr = FcIPLLR(model_config)
                model_config.scheduler = None
                ip = StandardFCIP(model_config)

                ipllr.copy_initial_params_from_model(ip, check_model=True)
                ipllr.initialize_params()
                ip.train()
                ipllr.train()

                for i, batch in enumerate(batches):
                    x, y = batch

                    # set gradients to 0
                    ipllr.optimizer.zero_grad()
                    ip.optimizer.zero_grad()

                    # outputs at initialization
                    y_hat_llr = ipllr.forward(x)
                    loss_ipllr = ipllr.loss(y_hat_llr, y)

                    y_hat_ip = ip.forward(x)
                    loss_ip = ip.loss(y_hat_ip, y)

                    # test outputs equality
                    torch.testing.assert_allclose(y_hat_llr, y_hat_ip, rtol=1e-6, atol=1e-6)

                    # gradients at initialization
                    loss_ipllr.backward()
                    loss_ip.backward()

                    # don't take any optimizer or scheduler step and test gradients equality

                    # input layer
                    torch.testing.assert_allclose(ipllr.input_layer.weight.grad, ip.input_layer.weight.grad,
                                                  rtol=1e-6, atol=1e-6)
                    torch.testing.assert_allclose(ipllr.input_layer.bias.grad, ip.input_layer.bias.grad,
                                                  rtol=1e-6, atol=1e-6)

                    # intermediate layers
                    for l in range(2, ip.n_layers):
                        ipllr_layer = getattr(ipllr.intermediate_layers, "layer_{:,}_intermediate".format(l))
                        ip_layer = getattr(ip.intermediate_layers, "layer_{:,}_intermediate".format(l))

                        torch.testing.assert_allclose(ipllr_layer.weight.grad, ip_layer.weight.grad,
                                                      rtol=1e-6, atol=1e-6)
                        torch.testing.assert_allclose(ipllr_layer.bias.grad, ip_layer.bias.grad,
                                                      rtol=1e-6, atol=1e-6)

                    # output layer
                    torch.testing.assert_allclose(ipllr.output_layer.weight.grad, ip.output_layer.weight.grad,
                                                  rtol=1e-6, atol=1e-6)
                    torch.testing.assert_allclose(ipllr.output_layer.bias.grad, ip.output_layer.bias.grad,
                                                  rtol=1e-6, atol=1e-6)

    def test_ipllr_vs_muP(self):
        n_warmup_steps = int(10e5)
        n_samples = 200
        batch_size = 1
        n_batches = math.ceil(n_samples / batch_size)
        output_size = 1

        xs = torch.rand(size=(n_samples, self.input_size))
        ys = torch.rand(size=(n_samples, output_size))
        # ys = torch.randint(high=10, size=(n_samples,))
        batches = [(xs[i * batch_size: (i+1) * batch_size, :], ys[i * batch_size: (i+1) * batch_size, :])
                   for i in range(n_batches)]

        Ls = [2, 3, 4]
        base_lrs = [1., 0.1, 0.01]
        width = 1024
        activation = 'relu'
        bias = False

        for L in Ls:
            for base_lr in base_lrs:
                model_config = deepcopy(self.base_model_config)
                model_config.scheduler.params['n_warmup_steps'] = n_warmup_steps
                model_config.architecture['n_layers'] = L + 1
                model_config.architecture['output_size'] = output_size
                model_config.architecture['width'] = width
                model_config.architecture['bias'] = bias
                model_config.activation.name = activation
                model_config.loss.name = 'mse'
                model_config.optimizer.params['lr'] = base_lr

                ipllr = FcIPLLR(model_config)
                model_config.scheduler = None
                muP = FCmuP(model_config)
                ntk = FCNTK(model_config)

                ipllr.copy_initial_params_from_model(ntk, check_model=True)
                ipllr.initialize_params()

                muP.copy_initial_params_from_model(ntk, check_model=True)
                muP.initialize_params()

                self.assertEqual(ipllr.base_lr, base_lr)
                self.assertEqual(ipllr.scheduler.base_lr, base_lr)
                self.assertEqual(muP.base_lr, base_lr)
                self.assertEqual(ntk.base_lr, base_lr)

                # set all input biases to 0
                with torch.no_grad():
                    ipllr.input_layer.bias.data.fill_(0.)
                    muP.input_layer.bias.data.fill_(0.)
                    ntk.input_layer.bias.data.fill_(0.)

                ipllr.train()
                muP.train()
                ntk.train()

                chis_ipllr = []
                chis_muP = []
                xs_L_ntk = []
                for i, batch in enumerate(batches):
                    chi_ipllr, chi_muP, x_L_ntk = \
                        self.test_first_forward_backward_ip_muP(batch, ipllr, muP, ntk, width, L)
                    chis_ipllr.append(chi_ipllr)
                    chis_muP.append(chi_muP)
                    xs_L_ntk.append(x_L_ntk.mean().item())

                with torch.no_grad():
                    print('input means :', xs.mean().item())
                    print('mean x_L_ntk :', np.mean(xs_L_ntk))
                    print('mean chi_muP :', np.mean(chis_muP))
                    print('mean chi_ipllr :', np.mean(chis_ipllr))
                    print('')

    def test_first_forward_backward_ip_muP(self, batch, ipllr, muP, ntk, width, L):
        x, y = batch
        with torch.no_grad():
            x_0 = x.clone().detach()

        # set gradients to 0
        ipllr.optimizer.zero_grad()
        muP.optimizer.zero_grad()
        ntk.optimizer.zero_grad()

        # outputs at initialization
        x_1_ipllr = ipllr.activation((ipllr.width ** (-ipllr.a[0])) * ipllr.input_layer.forward(x))
        x_1_muP = muP.activation((muP.width ** (-muP.a[0])) * muP.input_layer.forward(x))
        x_1_ntk = ntk.activation((ntk.width ** (-ntk.a[0])) * ntk.input_layer.forward(x))

        with torch.no_grad():
            torch.testing.assert_allclose(x_1_ipllr, x_1_ntk, rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(x_1_muP, x_1_ntk, rtol=1e-6, atol=1e-6)

        x = x_1_ipllr
        xs_ipllr = []
        for l, layer in enumerate(ipllr.intermediate_layers):
            h = (ipllr.width ** (-ipllr.a[l + 1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = ipllr.activation(h)  # x_l, l-th layer activations
            xs_ipllr.append(x)
        x_L_ipllr = x

        x = x_1_muP
        xs_muP = []
        for l, layer in enumerate(muP.intermediate_layers):
            h = (muP.width ** (-muP.a[l + 1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = muP.activation(h)  # x_l, l-th layer activations
            xs_muP.append(x)
        x_L_muP = x

        x = x_1_ntk
        xs_ntk = []
        for l, layer in enumerate(ntk.intermediate_layers):
            h = (ntk.width ** (-ntk.a[l + 1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = ntk.activation(h)  # x_l, l-th layer activations
            xs_ntk.append(x)
        x_L_ntk = x

        with torch.no_grad():
            for l in range(2, L+1):
                torch.testing.assert_allclose(xs_ipllr[l-2], (width ** (-(l-1)/2)) * xs_ntk[l-2], rtol=1e-6, atol=1e-6)
                torch.testing.assert_allclose(xs_muP[l-2], xs_ntk[l-2], rtol=1e-6, atol=1e-6)

            torch.testing.assert_allclose(x_L_ipllr, (width ** (-(L - 1) / 2)) * x_L_ntk, rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(x_L_muP, x_L_ntk, rtol=1e-6, atol=1e-6)

        y_hat_ipllr_debug = (ipllr.width ** (-ipllr.a[L])) * ipllr.output_layer.forward(x_L_ipllr)
        y_hat_muP_debug = (muP.width ** (-muP.a[L])) * muP.output_layer.forward(x_L_muP)
        y_hat_ntk_debug = (ntk.width ** (-ntk.a[L])) * ntk.output_layer.forward(x_L_ntk)

        y_hat_ipllr = ipllr.forward(x_0)
        y_hat_ipllr.retain_grad()
        loss_ipllr = ipllr.loss(y_hat_ipllr, y)

        y_hat_mup = muP.forward(x_0)
        y_hat_mup.retain_grad()
        loss_mup = muP.loss(y_hat_mup, y)

        y_hat_ntk = ntk.forward(x_0)
        y_hat_ntk.retain_grad()
        loss_ntk = ntk.loss(y_hat_ntk, y)

        with torch.no_grad():
            # test debug y_hats
            torch.testing.assert_allclose(y_hat_ipllr_debug, y_hat_ipllr, rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(y_hat_muP_debug, y_hat_mup, rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(y_hat_ntk_debug, y_hat_ntk, rtol=1e-6, atol=1e-6)

            # test outputs equality
            torch.testing.assert_allclose(y_hat_mup, (width ** (-1 / 2)) * y_hat_ntk, rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(y_hat_ipllr, (width ** (-L / 2)) * y_hat_ntk, rtol=1e-6, atol=1e-6)

        # gradients at initialization
        loss_ipllr.backward()
        loss_mup.backward()
        loss_ntk.backward()

        with torch.no_grad():
            # define chi_s
            chi_ipllr = y_hat_ipllr.grad.item()
            chi_muP = y_hat_mup.grad.item()

            # test output layer gradient (w.r.t to the learnable weights)
            torch.testing.assert_allclose(ipllr.output_layer.weight.grad, (1 / width) * chi_ipllr * x_L_ipllr,
                                          rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(ipllr.output_layer.weight.grad, (width ** (-(L+1)/2)) * chi_ipllr * x_L_ntk,
                                          rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(muP.output_layer.weight.grad, (width ** (-1/2)) * chi_muP * x_L_muP,
                                          rtol=1e-6, atol=1e-6)
            torch.testing.assert_allclose(muP.output_layer.weight.grad, (width ** (-1/2)) * chi_muP * x_L_ntk,
                                          rtol=1e-6, atol=1e-6)

            loss_derivative_ratio = chi_muP / chi_ipllr

            # test gradients
            # Delta_W = -eta * m^{-a + c} grad_w (Loss)

            # output layer grads
            Delta_W_ipllr = ipllr.base_lr * (width ** (- ipllr.a[L])) * ipllr.scheduler.initial_lrs[L] * \
                            ipllr.output_layer.weight.grad
            Delta_W_muP = muP.base_lr * (width ** (- muP.a[L])) * muP.output_layer.weight.grad

            torch.testing.assert_allclose(loss_derivative_ratio * Delta_W_ipllr, Delta_W_muP, rtol=1e-5, atol=1e-5)

            # intermediate layer grads
            for l in range(2, L+1):
                ipllr_layer = getattr(ipllr.intermediate_layers, "layer_{:,}_intermediate".format(l))
                muP_layer = getattr(muP.intermediate_layers, "layer_{:,}_intermediate".format(l))

                Delta_W_ipllr = ipllr.base_lr * (width ** (- ipllr.a[l-1])) * ipllr.scheduler.initial_lrs[l-1] * \
                                ipllr_layer.weight.grad
                Delta_W_muP = muP.base_lr * (width ** (- muP.a[l-1])) * muP_layer.weight.grad

                torch.testing.assert_allclose(loss_derivative_ratio * Delta_W_ipllr, Delta_W_muP, rtol=1e-5, atol=1e-5)

            # input layer grads
            Delta_W_ipllr = ipllr.base_lr * (width ** (- ipllr.a[0])) * ipllr.scheduler.initial_lrs[0] * \
                            ipllr.input_layer.weight.grad
            Delta_W_muP = muP.base_lr * (width ** (- muP.a[0])) * muP.input_layer.weight.grad

            torch.testing.assert_allclose(loss_derivative_ratio * Delta_W_ipllr, Delta_W_muP, rtol=1e-5, atol=1e-5)

        return chi_ipllr, chi_muP, x_L_ntk

    def test_first_weight_update_ipllr(self):
        n_warmup_steps = int(10e5)
        n_samples = 200
        batch_size = 1
        n_batches = math.ceil(n_samples / batch_size)
        output_size = 1

        xs = torch.rand(size=(n_samples, self.input_size))
        ys = torch.rand(size=(n_samples, output_size))
        # ys = torch.randint(high=10, size=(n_samples,))
        batches = [(xs[i * batch_size: (i + 1) * batch_size, :], ys[i * batch_size: (i + 1) * batch_size, :])
                   for i in range(n_batches)]

        Ls = [2, 3, 4]
        base_lrs = [1., 0.1, 0.01]
        # Ls = [3]
        # base_lrs = [0.01]
        width = 1024
        activation = 'relu'
        bias = False

        for L in Ls:
            for base_lr in base_lrs:
                model_config = deepcopy(self.base_model_config)
                model_config.scheduler.params['n_warmup_steps'] = n_warmup_steps
                model_config.architecture['n_layers'] = L + 1
                model_config.architecture['output_size'] = output_size
                model_config.architecture['width'] = width
                model_config.architecture['bias'] = bias
                model_config.activation.name = activation
                model_config.loss.name = 'mse'
                model_config.optimizer.params['lr'] = base_lr

                ipllr = FcIPLLR(model_config)
                model_config.scheduler = None
                muP = FCmuP(model_config)
                ntk = FCNTK(model_config)

                ipllr.copy_initial_params_from_model(ntk, check_model=True)
                ipllr.initialize_params()

                muP.copy_initial_params_from_model(ntk, check_model=True)
                muP.initialize_params()

                self.assertEqual(ipllr.base_lr, base_lr)
                self.assertEqual(ipllr.scheduler.base_lr, base_lr)
                self.assertEqual(muP.base_lr, base_lr)
                self.assertEqual(ntk.base_lr, base_lr)

                # set all input biases to 0
                with torch.no_grad():
                    ipllr.input_layer.bias.data.fill_(0.)
                    muP.input_layer.bias.data.fill_(0.)
                    ntk.input_layer.bias.data.fill_(0.)

                ipllr_init = deepcopy(ipllr)
                muP_init = deepcopy(muP)
                ntk_init = deepcopy(ntk)

                ipllr.train()
                muP.train()
                ntk.train()

                for idx, batch in enumerate(batches):
                    if idx == 0:
                        loss_derivative_ratio = self._test_first_weight_updates(idx, batch, ipllr, muP, ntk, ipllr_init,
                                                                                muP_init, ntk_init, width, L, base_lr,
                                                                                loss_derivative_ratio=0)
                    else:
                        _ = self._test_first_weight_updates(idx, batch, ipllr, muP, ntk, ipllr_init, muP_init, ntk_init,
                                                            width, L, base_lr,
                                                            loss_derivative_ratio=loss_derivative_ratio)

    def _test_first_weight_updates(self, idx, batch, ipllr, muP, ntk, ipllr_init, muP_init, ntk_init, width, L,
                                   base_lr, loss_derivative_ratio):
        x, y = batch
        with torch.no_grad():
            x_0 = x.clone().detach()

        # set gradients to 0
        if idx == 0:
            ipllr.optimizer.zero_grad()
            muP.optimizer.zero_grad()
            ntk.optimizer.zero_grad()

        y_hat_ipllr = ipllr.forward(x_0)
        y_hat_ipllr.retain_grad()
        loss_ipllr = ipllr.loss(y_hat_ipllr, y)

        y_hat_mup = muP.forward(x_0)
        y_hat_mup.retain_grad()
        loss_mup = muP.loss(y_hat_mup, y)

        y_hat_ntk = ntk.forward(x_0)
        y_hat_ntk.retain_grad()
        loss_ntk = ntk.loss(y_hat_ntk, y)

        if idx == 0:
            # gradients at initialization
            loss_ipllr.backward()
            loss_mup.backward()
            loss_ntk.backward()

            # do the first update
            ipllr.optimizer.step()
            muP.optimizer.step()
            ntk.optimizer.step()

            # define chi_s
            chi_ipllr = y_hat_ipllr.grad.item()
            chi_muP = y_hat_mup.grad.item()
            loss_derivative_ratio = chi_muP / chi_ipllr

        else:
            with torch.no_grad():
                # gradients for the weights should hold the initial gradients as the backward pass is not computed for
                # idx >= 1

                # input layer update
                Delta_w_ipllr = ipllr.input_layer.weight.data - ipllr_init.input_layer.weight.data
                Delta_w_muP = muP.input_layer.weight.data - muP_init.input_layer.weight.data
                torch.testing.assert_allclose(Delta_w_ipllr,
                                              - base_lr * (width ** ((L+1) / 2)) * ipllr.input_layer.weight.grad,
                                              rtol=1e-5, atol=1e-5)
                torch.testing.assert_allclose(Delta_w_muP,
                                              - base_lr * muP.input_layer.weight.grad,
                                              rtol=1e-5, atol=1e-5)
                torch.testing.assert_allclose(loss_derivative_ratio * Delta_w_ipllr, (width ** 0.5) * Delta_w_muP,
                                              rtol=1e-5, atol=1e-5)

                # intermediate layer grads
                for l in range(2, L + 1):
                    layer_key = "layer_{:,}_intermediate".format(l)
                    ipllr_layer = getattr(ipllr.intermediate_layers, layer_key)
                    muP_layer = getattr(muP.intermediate_layers, layer_key)
                    ipllr_init_layer = getattr(ipllr_init.intermediate_layers, layer_key)
                    muP_init_layer = getattr(muP_init.intermediate_layers, layer_key)

                    Delta_w_ipllr = ipllr_layer.weight.data - ipllr_init_layer.weight.data
                    Delta_w_muP = muP_layer.weight.data - muP_init_layer.weight.data

                    torch.testing.assert_allclose(Delta_w_ipllr,
                                                  - base_lr * (width ** ((L + 2) / 2)) * ipllr_layer.weight.grad,
                                                  rtol=1e-5, atol=1e-5)
                    torch.testing.assert_allclose(Delta_w_muP,
                                                  - base_lr * muP_layer.weight.grad,
                                                  rtol=1e-5, atol=1e-5)
                    torch.testing.assert_allclose(loss_derivative_ratio * (1 / width) * Delta_w_ipllr, Delta_w_muP,
                                                  rtol=1e-5, atol=1e-5)

                # output layer update
                Delta_w_ipllr = ipllr.output_layer.weight.data - ipllr_init.output_layer.weight.data
                Delta_w_muP = muP.output_layer.weight.data - muP_init.output_layer.weight.data
                torch.testing.assert_allclose(Delta_w_ipllr,
                                              - base_lr * (width ** ((L+1) / 2)) * ipllr.output_layer.weight.grad,
                                              rtol=1e-5, atol=1e-5)
                torch.testing.assert_allclose(Delta_w_muP,
                                              - base_lr * muP.output_layer.weight.grad,
                                              rtol=1e-5, atol=1e-5)
                torch.testing.assert_allclose(loss_derivative_ratio * (1 / width) * Delta_w_ipllr,
                                              (width ** (-1/2)) * Delta_w_muP,
                                              rtol=1e-5, atol=1e-5)

        return loss_derivative_ratio

    def test_second_forward_scales(self):
        n_warmup_steps = int(10e5)
        n_samples = 200
        batch_size = 1
        n_batches = math.ceil(n_samples / batch_size)
        output_size = 1

        xs = torch.rand(size=(n_samples, self.input_size))
        ys = torch.rand(size=(n_samples, output_size))
        # ys = torch.randint(high=10, size=(n_samples,))
        batches = [(xs[i * batch_size: (i + 1) * batch_size, :], ys[i * batch_size: (i + 1) * batch_size, :])
                   for i in range(n_batches)]

        Ls = [3, 4]
        base_lrs = [0.01, 0.1, 1.0]
        width = 1024
        activation = 'relu'
        bias = False

        for L in Ls:
            for base_lr in base_lrs:
                model_config = deepcopy(self.base_model_config)
                model_config.scheduler.params['n_warmup_steps'] = n_warmup_steps
                model_config.architecture['n_layers'] = L + 1
                model_config.architecture['output_size'] = output_size
                model_config.architecture['width'] = width
                model_config.architecture['bias'] = bias
                model_config.activation.name = activation
                model_config.loss.name = 'mse'
                model_config.optimizer.params['lr'] = base_lr

                ipllr = FcIPLLR(model_config)
                model_config.scheduler = None
                muP = FCmuP(model_config)
                ntk = FCNTK(model_config)

                ipllr.copy_initial_params_from_model(ntk, check_model=True)
                ipllr.initialize_params()

                muP.copy_initial_params_from_model(ntk, check_model=True)
                muP.initialize_params()

                self.assertEqual(ipllr.base_lr, base_lr)
                self.assertEqual(ipllr.scheduler.base_lr, base_lr)
                self.assertEqual(muP.base_lr, base_lr)
                self.assertEqual(ntk.base_lr, base_lr)

                # set all input biases to 0
                with torch.no_grad():
                    ipllr.input_layer.bias.data.fill_(0.)
                    muP.input_layer.bias.data.fill_(0.)
                    ntk.input_layer.bias.data.fill_(0.)

                ipllr_init = deepcopy(ipllr)
                muP_init = deepcopy(muP)
                ntk_init = deepcopy(ntk)

                ipllr.train()
                muP.train()
                ntk.train()
                
                # first train the models for one step
                x, y = batches[0]
                self._train_models_one_step(ipllr, muP, x, y)

                ipllr.eval()
                muP.eval()
                ntk.eval()

                contributions_df = pd.DataFrame(columns=['model', 'layer', 'init', 'update', 'total', 'id'])
                contributions_df.loc[:, ['init', 'update', 'total', 'id']] = \
                    contributions_df.loc[:, ['init', 'update', 'total', 'id']].astype(float)

                idx = self._compute_contributions(contributions_df, 'ipllr', ipllr, ipllr_init, batches, idx=0)
                _ = self._compute_contributions(contributions_df, 'muP', muP, muP_init, batches, idx)

                ipllr_contributions = contributions_df.loc[contributions_df.model == 'ipllr', :]
                muP_contributions = contributions_df.loc[contributions_df.model == 'muP', :]

                print('---- For L = {:,} and base_lr = {} ----'.format(L, base_lr))

                print('ipllr contributions per layer : ')
                print(ipllr_contributions.groupby(by='layer')[['init', 'update', 'total']].mean())
                print('')

                print('muP contributions per layer : ')
                print(muP_contributions.groupby(by='layer')[['init', 'update', 'total']].mean())
                print('\n\n')

    @staticmethod
    def _train_models_one_step(ipllr, muP, x, y):
        # set gradients to 0
        ipllr.optimizer.zero_grad()
        muP.optimizer.zero_grad()

        # outputs at initialization
        y_hat_llr = ipllr.forward(x)
        y_hat_llr.retain_grad()
        loss_ipllr = ipllr.loss(y_hat_llr, y)

        y_hat_muP = muP.forward(x)
        y_hat_muP.retain_grad()
        loss_muP = muP.loss(y_hat_muP, y)

        # gradients at initialization
        loss_ipllr.backward()
        loss_muP.backward()

        print('input mean in training: ', x.mean().item())
        print('loss derivative for IP-LLR :', y_hat_llr.grad.item())
        print('loss derivative for muP :', y_hat_muP.grad.item())
        print('')

        # first weight update
        ipllr.optimizer.step()
        muP.optimizer.step()

    @staticmethod
    def _compute_contributions(contributions_df, model_name, model, model_init, batches, idx):
        L = model.n_layers - 1
        with torch.no_grad():
            # don't use any bias, even in the first layer
            model.input_layer.bias.data.fill_(0.)

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
                                              rtol=1e-5, atol=1e-5)

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

                    init_contrib = model_init.layer_scales[l-1] * init_layer.forward(x)
                    update_contrib = model.layer_scales[l-1] * x.matmul(Delta_w.t())
                    total_contrib = model.layer_scales[l-1] * layer.forward(x)

                    torch.testing.assert_allclose(init_contrib + update_contrib, total_contrib,
                                                  rtol=1e-4, atol=1e-4)

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
                                              rtol=1e-5, atol=1e-5)

                contributions_df.loc[idx, ['model', 'layer', 'init', 'update', 'total', 'id']] = \
                    [model_name, L+1, init_contrib.mean().item(), update_contrib.mean().item(),
                     total_contrib.mean().item(), i]
                idx += 1

                y_hat_debug = total_contrib
                y_hat = model.forward(x_0)

                torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

        return idx


if __name__ == '__main__':
    unittest.main()
