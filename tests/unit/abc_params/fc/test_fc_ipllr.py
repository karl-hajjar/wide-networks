import unittest
import os
from copy import deepcopy
import torch

from utils.tools import read_yaml
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.fc_ipllr import FcIPLLR
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


if __name__ == '__main__':
    unittest.main()
