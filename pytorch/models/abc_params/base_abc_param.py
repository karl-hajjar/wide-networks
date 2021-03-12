import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import List
import logging
import math
from collections.abc import Iterable

from pytorch.models.base_model import BaseModel
from utils.nn import smoothed_exp_margin, smoothed_logistic_margin
from pytorch.initializers import INIT_DICT


class BaseABCParam(BaseModel):
    """
    A base class implementing the general skeleton of an abc-parameterization of a fully-connected network. For more
    details on abc-parameterizations see https://arxiv.org/abs/2011.14522. Anything that is architecture specific is
    left out of this class and has to be implemented in the child classes.
    """

    def __init__(self, config, a: List[float], b: List[float], c: [List[float], float], width: int = None):
        """

        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param a: list of floats. The layer scales. The pre-activations of layer l will be scaled by m^{-a[l]}.
        :param b: list of floats. The initialization scales. The initialization of layer l will be scaled by m^{-b[l]}
        :param c: float or list of floats. The learning rate scales. The learning rate of layer l will be scaled by
        m^{-c[l]} if c is iterable, and by m^{-c} otherwise.
        """
        self._set_width(config.architecture, width)  # set width m and check that it is not None
        self._set_n_layers(config.architecture)  # set n_layers L+1 and check that it is not None
        self._set_bias(config.architecture)  # defines self.bias based on the config
        self._set_var(config.initializer)  # defines self.std based on the config

        self._check_scales(a, b, c)

        self.a = a  # layer scales
        self.b = b  # init scales
        if not (isinstance(c, Iterable)):
            c = [c] * self.n_layers
        elif len(c) == 1:
            c = [c[0]] * self.n_layers
        self.c = c  # learning rate scales

        self.layer_scales = [self.width ** (-a[l]) for l in range(self.n_layers)]
        self.init_scales = [self.width ** (-b[l]) for l in range(self.n_layers)]
        self.lr_scales = [self.width ** (-c[l]) for l in range(self.n_layers)]

        # create optimizer, loss, activation, normalization
        super().__init__(config)

    def _set_width(self, config, width):
        """
        `width` is the `m`used in the maths
        :param config:
        :param width:
        :return:
        """
        if ("width" in config.keys()) and (config["width"] is not None):
            logging.info('Initializing width from config')
            self.width = config["width"]  # m
        elif width is not None:
            logging.info('Initializing width from argument')
            self.width = width  # m
        else:
            raise ValueError("Both the config and argument widths were None")

    def _set_n_layers(self, config):
        """
        `n_layers` is the `L+1`used in the maths
        :param config:
        :return:
        """
        if ("n_layers" in config.keys()) and (config["n_layers"] is not None):
            self.n_layers = config["n_layers"]  # L + 1
        else:
            raise ValueError("`n_layers` was None in the config")

    def _set_bias(self, config):
        if ("bias" in config.keys()) and (config["bias"] is not None):
            self.bias = config["bias"]
        else:
            self.bias = False

    def _set_var(self, config):
        if ("var" in config.params.keys()) and (config.params["var"] is not None):
            self.var = config.params["var"]
        else:
            self.var = 1.0

    def _check_scales(self, a, b, c):
        if len(a) != self.n_layers:
            raise ValueError("Layer scales `a` had {:,} elements but n_layers is {:,}")
        if len(b) != self.n_layers:
            raise ValueError("Init scales `b` had {:,} elements but n_layers is {:,}")
        if (isinstance(c, Iterable)) and (len(c) >= 2) and (len(c) != self.n_layers):
            raise ValueError("learning rate scales `c` had {:,} elements but n_layers is {:,}")

    def _build_model(self, config):
        self.input_layer = torch.nn.Module()
        self.intermediate_layers = torch.nn.ModuleList()
        self.output_layer = torch.nn.Module()

    def set_layer_scales_from_a(self):
        """
        To use in case we want to change the values of abc during the course of training.
        :return:
        """
        self.layer_scales = [self.width ** (-self.a[l]) for l in range(self.n_layers)]

    def initialize_params(self, init_config=None):
        """
        Initialize layer l matrix with m^{-b[l]} U^l where U^l_{ij} ~ N(0, 1) are iid (over i,j) Gaussians, and
        similarly layer l bias with m^{-b[l]} v^l where v^l_{i} ~ N(0, 1) are iid (over i) Gaussians.
        :param init_config:
        :return:
        """
        self._generate_standard_gaussians(math.sqrt(self.var))
        # self._generate_standard_gaussians(std=1.0)
        with torch.no_grad():
            # weights
            self.input_layer.weight.data.copy_(self.init_scales[0] * self.U[0].data)

            for l, layer in enumerate(self.intermediate_layers):
                layer.weight.data.copy_(self.init_scales[l+1] * self.U[l + 1].data)

            self.output_layer.weight.data.copy_(self.init_scales[self.n_layers - 1] * self.U[self.n_layers - 1].data)

            # biases
            if hasattr(self.input_layer, "bias"):
                self.input_layer.bias.data.copy_(self.init_scales[0] * self.v[0].data)

            if self.bias:
                for l, layer in enumerate(self.intermediate_layers):
                    layer.bias.data.copy_(self.init_scales[l+1] * self.v[l + 1].data)

                self.output_layer.bias.copy_(self.init_scales[self.n_layers - 1] * self.v[self.n_layers-1].data)

    def _generate_standard_gaussians(self, std=math.sqrt(2.0)):
        self.U = [torch.normal(mean=0, std=std, size=self.input_layer.weight.size(), requires_grad=False)]
        self.U += [torch.normal(mean=0, std=std, size=(self.width, self.width), requires_grad=False)
                   for _ in self.intermediate_layers]
        self.U.append(torch.normal(mean=0, std=1.0, size=self.output_layer.weight.size(), requires_grad=False))

        if hasattr(self.input_layer, "bias"):
            self.v = [torch.normal(mean=0, std=std, size=self.input_layer.bias.size(), requires_grad=False)]

        if self.bias:
            self.v += [torch.normal(mean=0, std=std, size=layer.bias.size(), requires_grad=False)
                       for layer in self.intermediate_layers]
            self.v.append(torch.normal(mean=0, std=1.0, size=self.output_layer.bias.size(), requires_grad=False))

    def copy_initial_params_from_model(self, model, check_model=False):
        """
        Set the matrices in self.U and biases in self.v from those of model if sizes match.
        :param model: an other abc-parameterization with compatible sizes for the weights and biases.
        :return:
        """
        if check_model:  # there is a check anyways in copy_initial_params_from_params_list
            self._check_model_compatible(model)
        v = model.v if hasattr(model, 'v') else None
        self.copy_initial_params_from_params_list(model.U, v)

    def copy_initial_params_from_params_list(self, U, v):
        self._check_params_compatible(U, v)
        with torch.no_grad():
            for l in range(self.n_layers):
                self.U[l].data.copy_(U[l].detach().data)
            self.v[0].data.copy_(v[0].detach().data)
            if self.bias:
                for l in range(1, self.n_layers):
                    self.v[l].data.copy_(v[l].detach().data)

    def _check_model_compatible(self, model):
        if not isinstance(model, BaseABCParam):
            raise TypeError("`model` is not an instance of BaseABCParam")
        if model.bias != self.bias:
            raise ValueError("`model` has bias set to {} while self has bias set to {}".format(model.bias, self.bias))
        if model.width != self.width:
            raise ValueError("`model` has width {:,} while self has width {:,}".format(model.width, self.width))
        if model.n_layers != self.n_layers:
            raise ValueError("`model` has {:,} layers while self has {:,} layers".format(model.n_layers, self.n_layers))

    def _check_params_compatible(self, U, v):
        if (len(v) != len(self.v)) or (len(U) != len(self.U)):
            raise ValueError("U or v did not match the length of self.U or self.v")
        for l in range(self.n_layers):
            if U[l].size() != self.U[l].size():
                raise ValueError("U[l] was of size {} whereas self.U[l] was of size {}".format(U[l].size(),
                                                                                               self.U[l].size()))
            if v[l].size() != self.v[l].size():
                raise ValueError("v[l] was of size {} whereas self.v[l] was of size {}".format(v[l].size(),
                                                                                               self.v[l].size()))
    def forward(self, x):
        # all about optimization with Lightning can be found here (e.g. how to define a particular optim step) :
        # https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        h = (self.width ** (-self.a[0])) * self.input_layer.forward(x)  # h_0, first layer pre-activations
        x = self.activation(h)  # x_0, first layer activations

        for l, layer in enumerate(self.intermediate_layers):  # L-1 intermediate layers
            h = (self.width ** (-self.a[l+1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = self.activation(h)  # x_l, l-th layer activations

        return (self.width ** (-self.a[self.n_layers-1])) * self.output_layer.forward(x)  # f(x)

    def predict(self, x, mode='probas', from_logits=False):
        # if not from_logits:
        #     x = self.forward(x)
        # if mode == 'probas':
        #     return torch.sigmoid(x)
        # else:
        #     return x
        pass

    def train_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def val_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def test_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def training_step(self, batch, batch_nb):
        pass

    # TODO : Apparently 'training_epoch_end' is not compatible with newer versions of PyTorch Lightning and might have
    #  to be changed to some other name such as 'on_epoch_end' for compliance with versions >= 0.9.0.
    def training_epoch_end(self, outputs):
        pass

    def _evaluation_step(self, batch, batch_nb, mode: str):
        pass

    def _evaluation_epoch_end(self, outputs, mode: str):
        pass

    def configure_optimizers(self):
        pass
