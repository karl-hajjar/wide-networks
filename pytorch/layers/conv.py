import torch.nn as nn
from pytorch_lightning import LightningModule


class ConvBlock(LightningModule):
    """
    A class defining a block of convolutions with same size.
    """

    def __init__(self, n_layers, activation, norm, n_channels, kernel_size=3, stride=1, padding=2,
                 bias=True, activate_last=False):
        super(ConvBlock, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.norm = norm
        assert n_layers > 0, "n_layers argument must be > 0, but was {:,}".format(n_layers)
        # converting list of modules to sequence of arguments
        self.conv_block = nn.Sequential(*[self._conv_layer(n_channels, kernel_size, stride, padding,
                                                           activation=activation, bias=bias)
                                          for _ in range(n_layers - 1)])
        # last conv as a standalone because it might be without anny activation
        self.conv_block.add_module("simple_conv",
                                   self._conv_layer(n_channels, kernel_size, stride, padding, activation=activation,
                                                    bias=bias))
        if activate_last:
            self.conv_block.add_module("last_activation", activation)

    @staticmethod
    def _conv_layer(n_channels, kernel_size, stride, padding, activation=None, bias=True):
        if activation is not None:
            # from different research, it turns out applying the conv, norm and activation in the following order
            # enhances performance : conv -> norm -> activation.
            return nn.Sequential(activation,
                                 nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=bias))
        else:
            return nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        if self.norm is None:
            return self.conv_block(x)
        else:
            return self.conv_block(self.norm(x))
