import torch
import torch.nn as nn
import numpy as np

from .base_model import BaseModel
from pytorch.layers.residual import ResidualConnection
from pytorch.layers.conv import ConvBlock
from pytorch.layers.fully_connected import FullyConnectedBlock
from utils.nn import get_output_size_conv, get_same_padding_conv


class ResNet(BaseModel):
    """
    A class implementing a residual network architecture with convolutional blocks.
    """

    def __init__(self, config):
        self.n_blocks = config.architecture["n_blocks"]
        assert self.n_blocks > 0, "n_blocks argument must be > 0 but was {:,}".format(self.n_blocks)
        self.n_channels = config.architecture["n_channels"]

        # create optimizer, loss, activation, normalization
        super(ResNet, self).__init__(config)
        if self._name == "model":
            self._name = "ResNet"  # change default name "model" to "ResNet"
        self._name = self._name + str(self.n_blocks)

    def _build_model(self, config):
        # create residual tower to process input
        self._build_res_block(config)

        # create fully connected layers before output
        self._build_fc_block(config)

    def _build_res_block(self, config):
        input_size = config.architecture["input_size"]
        kernel_size = config.architecture["kernel_size"]
        stride = config.architecture["stride"]
        bias_conv = config.architecture["bias_conv"]

        # first convolution to set dimensions
        self.conv1 = nn.Conv2d(config.architecture["in_channels"], self.n_channels, stride=stride,
                               kernel_size=kernel_size, padding=0)

        # output of convolution is of size = (width − kernel_size + 2 * padding) // stride + 1, and we want to keep this
        # size constant in the resnet block, and thus use padding = [(w - 1) * s - w + k] // 2
        self.res_block_input_size = get_output_size_conv(input_size, kernel_size, stride, padding=0)
        self._check_size(input_size, input_size, config.architecture["in_channels"], self.res_block_input_size, self.res_block_input_size, self.conv1)

        padding = get_same_padding_conv(self.res_block_input_size, kernel_size, stride)
        self._check_padding_for_residual(padding, self.res_block_input_size, self.res_block_input_size, kernel_size,
                                         stride)

        # define convolution blocks
        self.conv_blocks = nn.ModuleList([ConvBlock(n_layers=config.architecture["n_conv_layers"],
                                                    activation=self.activation,
                                                    norm=self.norm(self.n_channels, **config.normalization.params),
                                                    n_channels=self.n_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=bias_conv)
                                          for _ in range(self.n_blocks)])

        # check output size
        self._check_size(self.res_block_input_size, self.res_block_input_size, self.n_channels,
                         self.res_block_input_size, self.res_block_input_size, self.conv_blocks[0])

        # define residual module
        self.residual_connection = ResidualConnection(activation=self.activation)

        # define last convolution reducing number of channels
        self.conv2 = nn.Sequential(self.norm(self.n_channels, **config.normalization.params),
                                   nn.Conv2d(in_channels=self.n_channels,
                                             out_channels=self.n_channels // 4,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=0,
                                             bias=bias_conv),
                                   self.activation)

        self.fc_block_input_size = get_output_size_conv(self.res_block_input_size, kernel_size, stride, padding=0)
        self._check_size(self.res_block_input_size, self.res_block_input_size, self.n_channels, self.fc_block_input_size,
                         self.fc_block_input_size, self.conv2)

    def _build_fc_block(self, config):
        fc_dim = config.architecture["fc_dim"]
        bias_fc = config.architecture["bias_fc"]
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=(self.n_channels // 4) * (self.fc_block_input_size ** 2), out_features=fc_dim,
                      bias=bias_fc))
        # one fully connected has already be defined to have the appropriate shape, and another one will be needed to
        # have the expected output shape
        if config.architecture["n_fc_layers"] > 2:
            self.fc_block.add_module("fc_block",
                                     FullyConnectedBlock(n_layers=config.architecture["n_fc_layers"] - 2,
                                                         activation=self.activation,
                                                         norm=None,
                                                         in_features=fc_dim,
                                                         out_features=fc_dim,
                                                         bias=bias_fc,
                                                         activate_last=True))
        self.fc_block.add_module("output_layer",
                                 nn.Linear(in_features=fc_dim, out_features=config.architecture["output_size"],
                                           bias=bias_fc))

    @staticmethod
    def _check_size(input_width, input_height, input_channels, expected_output_width, expected_output_height,
                    module: nn.Module):
        if module.training:
            module.eval()
        with torch.no_grad():
            dummy_input = torch.zeros((1, input_channels, input_width, input_height), requires_grad=False)
            dummy_output_size = module.forward(dummy_input).size()
            dummy_output_width = dummy_output_size[2]
            dummy_output_height = dummy_output_size[3]
        assert expected_output_width == dummy_output_width, \
               "actual and expected output widths do not, they are respectively : {:,} and {:,}". \
               format(dummy_output_width, expected_output_width)
        assert expected_output_height == dummy_output_height, \
               "actual and expected output heights do not, they are respectively : {:,} and {:,}". \
               format(dummy_output_height, expected_output_height)

    @staticmethod
    def _check_padding_for_residual(padding, input_width, input_height, kernel_size, stride):
        output_width = get_output_size_conv(input_width, kernel_size, stride, padding)
        assert output_width == input_width, \
               "input and output widths of residual blocks won't match, they are respectively : {:,} and {:,}".\
               format(input_width, output_width)
        output_height = get_output_size_conv(input_height, kernel_size, stride, padding)
        assert output_height == input_height, \
               "input and output heights of residual blocks won't match, they are respectively : {:,} and {:,}".\
               format(input_height, output_height)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.conv_blocks:
            x = self.residual_connection(x, block)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)  # flatten over all dimensions except batch size
        return self.fc_block(x)
