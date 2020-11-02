import unittest
import torch
import numpy as np

import utils.plot.initialization.network as net_init


class TestInitPlotsInitialization(unittest.TestCase):
    def setUp(self) -> None:
        self.d = 100  # dimension of the input data
        self.n = int(1e3)  # number of samples considered
        self.Ms = [50, 100, 500]
        self.Ls = [3, 5, 10]
        self.bias = False
        self.x = torch.rand((self.n, self.d))

    def test_init(self):
        scaling = 'mf'
        for m in self.Ms:
            for L in self.Ls:
                for init, init_params in [('uniform', {'a': -2.0, 'b': 2.0}), ('normal', {'mean': 0., 'std': 2.0})]:
                    net = net_init.FCNetwork(L=L, m=m, d=self.d, bias=self.bias, init=init, scaling=scaling,
                                             init_params=init_params)
                    self.assertDictEqual(net.init_params, init_params)

    def test_sqrt_init_scaling(self):
        scaling = 'mf'
        scale_init = 'sqrt'
        for m in self.Ms:
            for L in self.Ls:
                for init, init_params in [('uniform', {'a': -2.0, 'b': 2.0}), ('normal', {'mean': 0., 'std': 2.0})]:
                    expected_rescaled_init_params = net_init._rescale_init(init_params, scale_init=scale_init, scale=m)
                    net = net_init.FCNetwork(L=L, m=m, d=self.d, bias=self.bias, init=init, scaling=scaling,
                                             init_params=expected_rescaled_init_params)

                    actual_rescaled_init_params = net.init_params

                    msg = 'init_params: {}\n\nactual rescaled_init_params: {}\n\n expected init_params'.\
                        format(init_params, actual_rescaled_init_params, expected_rescaled_init_params)
                    self.assertDictEqual(expected_rescaled_init_params, actual_rescaled_init_params, msg=msg)

    def test_lin_init_scaling(self):
        scaling = 'mf'
        scale_init = 'lin'
        for m in self.Ms:
            for L in self.Ls:
                for init, init_params in [('uniform', {'a': -2.0, 'b': 2.0}), ('normal', {'mean': 0., 'std': 2.0})]:
                    expected_rescaled_init_params = net_init._rescale_init(init_params, scale_init=scale_init, scale=m)
                    net = net_init.FCNetwork(L=L, m=m, d=self.d, bias=self.bias, init=init, scaling=scaling,
                                             init_params=expected_rescaled_init_params)

                    actual_rescaled_init_params = net.init_params

                    msg = 'init_params: {}\n\nactual rescaled_init_params: {}\n\n expected init_params'.\
                        format(init_params, actual_rescaled_init_params, expected_rescaled_init_params)
                    self.assertDictEqual(expected_rescaled_init_params, actual_rescaled_init_params, msg=msg)


if __name__ == '__main__':
    unittest.main()
