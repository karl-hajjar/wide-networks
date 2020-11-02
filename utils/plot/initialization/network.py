import torch
import torch.nn as nn
import numpy as np

from pytorch.initializers import INIT_DICT


sigma = nn.ReLU()


class WideLayer(nn.Module):
    """
    A class implementing a standard fully-connected layer with ReLU activations in the MF-scaling: i.e. where the
    output of the layer is re-scaled by 1/m where m is the number of hidden units.
    """

    def __init__(self, scale, in_features, out_features, scaling='mf', activation=None, bias=True):
        super().__init__()
        if scaling == 'ntk':
            scale = np.sqrt(scale)
        elif scaling == 'mf':
            pass
        else:
            raise ValueError("scaling {} not implemented".format(scaling))
        self.scale = scale
        self.scaling = scaling
        if activation is not None:
            self.layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
                                       activation)
        else:
            self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        return self.scale * self.layer.forward(x)


class FCNetwork(nn.Module):
    """
    A class implementing a standard fully-connected network with ReLU activations in the MF-scaling.
    """

    def __init__(self, L, m, d=100, bias=False, init=None, scaling='mf', init_params=None):
        super().__init__()
        assert L >= 2, "m must be > 1 but was {:,}".format(m)
        self.L = L
        self.m = m
        self.scaling = scaling
        self.init_params = init_params

        self.layer_sizes = [(d, m)] + [(m, m)] * (L - 2) + [(m, 1)]
        self.biases = [True] + [bias] * (L - 1)
        self.activations = [sigma] * (L - 1) + [None]
        self.scales = [1.0] + [1.0 / m] * (L - 1)
        assert len(self.layer_sizes) == len(self.biases) == L

        self.layers = nn.Sequential(*[WideLayer(scale=self.scales[l],
                                                in_features=self.layer_sizes[l][0],
                                                out_features=self.layer_sizes[l][1],
                                                activation=self.activations[l],
                                                bias=self.biases[l],
                                                scaling=scaling) for l in range(L)])
        if init is not None:
            self.initialize_params(init, init_params)

    def initialize_params(self, init, init_params):
        initializer = INIT_DICT[init]
        for p in self.parameters():
            if p.dim() > 1:
                if init_params is not None:
                    initializer(p, **init_params)
                else:
                    initializer(p)
            else:  # initialize biases
                # TODO : to complete later
                pass

    def forward(self, x):
        return self.layers.forward(x)


def define_networks(L, ms, d, init, init_params, n_trials, bias=False, scaling='mf', scale_init=None,
                    training=False):
    nets = dict()
    for m in ms:
        nets_ = []
        if (scale_init is not None) and (init_params is not None):
            rescaled_init_params = _rescale_init(init_params, scale_init, m)
        else:
            rescaled_init_params = init_params
        for _ in range(n_trials):
            net = FCNetwork(L=L, m=m, d=d, bias=bias, init=init, scaling=scaling, init_params=rescaled_init_params)
            if training:
                net.train()
            else:
                net.eval()  # set validation mode to disable autograd and release memory
            nets_.append(net)
        nets[m] = nets_
    return nets


def _rescale_init(init_params, scale_init, scale):
    if scale_init == 'lin':
        rescaled_init_params = {key: scale * value for key, value in init_params.items()}
    elif scale_init == 'sqrt':
        rescaled_init_params = {key: np.sqrt(scale) * value for key, value in init_params.items()}
    else:
        raise ValueError("init scaling {} is not implemented".format(scale_init))
    return rescaled_init_params
