import unittest
import torch

from utils.plot.initialization.network import *


class TestInitPlotsNetClass(unittest.TestCase):
    def setUp(self) -> None:
        self.d = 100  # dimension of the input data
        self.n = int(1e3)  # number of samples considered
        self.Ms = [50, 100, 500]
        self.Ls = [3, 5, 10]
        self.bias = False
        self.x = torch.rand((self.n, self.d))

    def test_network_output(self):
        for m in self.Ms:
            for L in self.Ls:
                for scaling in ['mf', 'ntk']:
                    net = FCNetwork(L=L, m=m, d=self.d, bias=self.bias, init=None, scaling=scaling, init_params=None)
                    outputs = net(self.x)
                    self.assertIsInstance(outputs, torch.Tensor)
                    self.assertTrue(outputs.shape == torch.Size([self.n, 1]))

    def test_layers_outputs(self):
        for m in self.Ms:
            for L in self.Ls:
                for scaling in ['mf', 'ntk']:
                    net = FCNetwork(L=L, m=m, d=self.d, bias=self.bias, init=None, scaling=scaling, init_params=None)
                    a = self.x
                    for l in range(L-1):  # last layer is tested in test_network_output
                        a = net.layers[l].forward(a)
                        self.assertIsInstance(a, torch.Tensor)
                        self.assertTrue(a.shape == torch.Size([self.n, m]))


if __name__ == '__main__':
    unittest.main()
