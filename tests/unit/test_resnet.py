import unittest
import os
from copy import deepcopy
import torch

from pytorch.configs.model import ModelConfig
from pytorch.models.resnet import ResNet
from pytorch.models.base_model import BaseModel

RESOURCES_DIR = '../resources/'
BATCH_SIZE = 32
INPUT_SIZES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 50, 51, 53, 56, 100, 101, 102, 128]
CHANNELS = [1, 3, 5, 10]


class TestResNet(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(config_file=os.path.join(RESOURCES_DIR, 'resnet_config.yaml'))

    def test_base_model_init(self):
        base = BaseModel(self.config)

    def test_network_build(self):
        resnet = ResNet(self.config)
        print(resnet)

    def test_padding_with_mutiple_input_sizes(self):
        config = deepcopy(self.config)
        for size in INPUT_SIZES:
            config.architecture["input_size"] = size
            resnet = ResNet(self.config)
            print(resnet)

    def test_forward_no_batch(self):
        resnet = ResNet(self.config)  # define resnet model

        # batch_size = 1, n_input_channels = 1
        x = torch.randn((1, 1, self.config.architecture["input_size"], self.config.architecture["input_size"]),
                        requires_grad=False)
        out = resnet(x)
        self.assertTrue(out.shape == torch.Size((1, self.config.architecture["output_size"])))

    def test_forward_batch(self):
        resnet = ResNet(self.config)  # define resnet model

        # batch_size = BATCH_SIZE, n_input_channels = 1
        x = torch.randn((BATCH_SIZE, 1, self.config.architecture["input_size"], self.config.architecture["input_size"]),
                        requires_grad=False)
        out = resnet(x)
        self.assertTrue(out.shape == torch.Size((BATCH_SIZE, self.config.architecture["output_size"])))

    def test_forward_mutiple_channels(self):
        config = deepcopy(self.config)  # define resnet model
        for c in CHANNELS:
            config.architecture["in_channels"] = c
            resnet = ResNet(config)
            x = torch.randn((BATCH_SIZE, c, self.config.architecture["input_size"],
                             self.config.architecture["input_size"]), requires_grad=False)
            out = resnet(x)
            self.assertTrue(out.shape == torch.Size((BATCH_SIZE, self.config.architecture["output_size"])))


if __name__ == '__main__':
    unittest.main()
