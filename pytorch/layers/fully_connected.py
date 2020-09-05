import torch.nn as nn
from pytorch_lightning import LightningModule


class FullyConnectedBlock(LightningModule):
    """
    A class defining a block of fully connected layers with same size.
    """

    def __init__(self, n_layers, activation, norm, in_features, out_features, bias=True, activate_last=False):
        super(FullyConnectedBlock, self).__init__()
        self.norm = norm
        assert n_layers > 0, "n_layers argument must be > 0, but was {:,}".format(n_layers)
        # converting list of modules to sequence of arguments
        self.fc_block = nn.Sequential(*[self._fc_layer(in_features, out_features, activation=activation, bias=bias)
                                        for _ in range(n_layers - 1)])
        # last fully-connected as a standalone because it might be without anny activation
        self.fc_block.add_module("simple_fc",
                                 self._fc_layer(in_features, out_features, activation=activation, bias=bias))
        if activate_last:
            self.fc_block.add_module("last_activation", activation)

    @staticmethod
    def _fc_layer(in_features, out_features, activation=None, bias=True):
        if activation is not None:
            # from different research, it turns out applying the conv, norm and activation in the following order
            # enhances performance : conv -> norm -> activation. We keep a similar order for fully connected networks
            return nn.Sequential(activation,
                                 nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
        else:
            return nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        if self.norm is None:
            return self.fc_block(x)
        else:
            return self.fc_block(self.norm(x))
