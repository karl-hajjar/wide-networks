import torch
from typing import List
import logging
import math
from collections.abc import Iterable
import numpy as np
import pickle

from pytorch.models.base_model import BaseModel


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
        self._set_init_std(config.initializer, config.activation.name)  # defines self.std based on the config
        self._set_init_mean(config.initializer)  # defines self.init_mean based on the config

        self._check_scales(a, b, c)

        self.a = a  # layer scales
        self.b = b  # init scales
        if not (isinstance(c, Iterable)):
            c = [c] * self.n_layers
        elif len(c) == 1:
            c = [c[0]] * self.n_layers
        self.c = c  # learning rate scales

        self.set_layer_scales_from_a()  # sets self.layer_scales
        self.init_scales = self._set_scales_from_exponents(self.b)
        self.set_lr_scales_from_c()  # sets self.lr_scales

        self.d = config.architecture["input_size"]  # input dimension

        # create optimizer, loss, activation, normalization and initializes parameters
        super().__init__(config)

    def _set_width(self, config, width):
        """
        `width` is the `m`used in the maths
        :param config:
        :param width:
        :return:
        """
        logger = logging.getLogger()
        if ("width" in config.keys()) and (config["width"] is not None):
            logger.info('Initializing width from config')
            self.width = config["width"]  # m
        elif width is not None:
            logger.info('Initializing width from argument')
            self.width = width  # m
        else:
            raise ValueError("Both the config and argument widths were None")

    def _set_n_layers(self, config):
        """
        `n_layers` is the `L+1`used in the maths
        :param config:
        :return:
        """
        if not hasattr(self, "n_layers"):
            if ("n_layers" in config.keys()) and (config["n_layers"] is not None):
                self.n_layers = config["n_layers"]  # L + 1
            else:
                raise ValueError("`n_layers` was None in the config")

    def _set_bias(self, config):
        if ("bias" in config.keys()) and (config["bias"] is not None):
            self.bias = config["bias"]
            if self.bias == True:
                if ("scale_bias" in config.keys()) and (config["scale_bias"] is not None):
                    self.scale_bias = config["scale_bias"]
                else:
                    self.scale_bias = True  # default is to scale bias same as the weights
            else:
                self.scale_bias = True  # default is to scale bias same as the weights
        else:
            self.bias = False
            self.scale_bias = True  # default is to scale bias same as the weights

    def _set_init_std(self, config, activation=None):
        var = 2.0  # default value for the variance
        if activation is not None:
            if activation == 'relu':
                var = 2.0
            elif activation == 'gelu':
                var = 4.0
            elif activation in ['elu', 'tanh']:
                var = 1.0
        elif ("var" in config.params.keys()) and (config.params["var"] is not None):
            var = config.params["var"]

        self.std = math.sqrt(var)

    def _set_init_mean(self, config):
        if ("mean" in config.params.keys()) and (config.params["mean"] is not None):
            mean = config.params["mean"]
        else:
            mean = 0.0

        self.init_mean = mean

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

    def _set_optimizer(self, optimizer_config, params=None):
        if (optimizer_config.params is None) or ('lr' not in optimizer_config.params.keys()):
            raise ValueError("optimizer config must have a parameter `lr` in abc-parameterizations")
        else:
            self.base_lr = optimizer_config.params['lr']

        # Layer-wise learning rates : one parameter group per layer with its own lr
        # In most abc-parameterization the intermediate layers will all share the same learning rate
        param_groups = \
            [{'params': self.input_layer.parameters(), 'lr': self.base_lr * self.lr_scales[0]}] + \
            [{'params': layer.parameters(), 'lr': self.base_lr * self.lr_scales[l+1]}
             for l, layer in enumerate(self.intermediate_layers)] + \
            [{'params': self.output_layer.parameters(), 'lr': self.base_lr * self.lr_scales[self.n_layers - 1]}]

        super()._set_optimizer(optimizer_config, params=param_groups)

    def _get_opt_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]  # L+1 lrs, one for each layer

    def _set_scales_from_exponents(self, exponents):
        """
        Returns the list [m^{-q} for q in exponents].
        :param exponents: a list of floats or ints
        :return:
        """
        return [self.width ** (-exponents[l]) for l in range(self.n_layers)]

    def set_layer_scales_from_a(self):
        """
        To use in case we want to change the values of a during the course of training.
        :return:
        """
        self.layer_scales = self._set_scales_from_exponents(self.a)

    def set_lr_scales_from_c(self):
        """
        To use in case we want to change the values of c during the course of training.
        :return:
        """
        self.lr_scales = self._set_scales_from_exponents(self.c)

    def initialize_params(self, init_config=None):
        """
        Initialize layer l matrix with m^{-b[l]} U^l where U^l_{ij} ~ N(0, 1) are iid (over i,j) Gaussians, and
        similarly layer l bias with m^{-b[l]} v^l where v^l_{i} ~ N(0, 1) are iid (over i) Gaussians.
        :param init_config:
        :return:
        """
        self._generate_standard_gaussians(mean=self.init_mean, std=self.std)
        with torch.no_grad():
            # weights
            self.input_layer.weight.data.copy_(self.init_scales[0] * self.U[0].data)

            for l, layer in enumerate(self.intermediate_layers):
                layer.weight.data.copy_(self.init_scales[l+1] * self.U[l + 1].data)

            self.output_layer.weight.data.copy_(self.init_scales[self.n_layers - 1] * self.U[self.n_layers - 1].data)

            # biases
            if hasattr(self.input_layer, "bias"):
                if self.scale_bias:
                    self.input_layer.bias.data.copy_(self.init_scales[0] * self.v[0].data)
                else:
                    self.input_layer.bias.data.copy_(self.v[0].data)

            if self.bias:
                for l, layer in enumerate(self.intermediate_layers):
                    if self.scale_bias:
                        layer.bias.data.copy_(self.init_scales[l+1] * self.v[l + 1].data)
                    else:
                        layer.bias.data.copy_(self.v[l + 1].data)
                if self.scale_bias:
                    self.output_layer.bias.copy_(self.init_scales[self.n_layers - 1] * self.v[self.n_layers-1].data)
                else:
                    self.output_layer.bias.copy_(self.v[self.n_layers-1].data)

    def _generate_standard_gaussians(self, mean=0.0, std=math.sqrt(2.0)):
        """
        Generate Gaussian matrices U and vectors v without any scaling (i.e. the whose values do NOT depend on the width
        of the network) with a given std, if those matrices and vectors are not already defined.
        :param std:
        :return:
        """
        if not hasattr(self, "U") or (self.U is None):
            # normalize std of first layer to avoid problems with large input dimension d
            self.U = [torch.normal(mean=0, std=std/math.sqrt(self.d + 1), size=self.input_layer.weight.size(),
                                   requires_grad=False)]
            self.U += [torch.normal(mean=mean, std=std, size=(self.width, self.width), requires_grad=False)
                       for _ in self.intermediate_layers]
            self.U.append(torch.normal(mean=mean, std=1.0, size=self.output_layer.weight.size(), requires_grad=False))

        if not hasattr(self, "v") or (self.v is None):
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
        :param check_model: bool, whether or not to do a first check that the model provided is compatible in terms of
        architecture with the object contained in self (check number of layers and if there is a bias or not).
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
                raise ValueError("U[{}] was of size {} whereas self.U[{}] was of size {}".format(l,
                                                                                                 U[l].size(),
                                                                                                 l,
                                                                                                 self.U[l].size()))

        if v[0].size() != self.v[0].size():
            raise ValueError("v[0] was of size {} whereas self.v[0] was of size {}".format(v[0].size(),
                                                                                           self.v[0].size()))
        if self.bias:
            for l in range(1, self.n_layers):
                if v[l].size() != self.v[l].size():
                    raise ValueError("v[{}] was of size {} whereas self.v[{}] was of size {}".format(l,
                                                                                                     v[l].size(),
                                                                                                     l,
                                                                                                     self.v[l].size()))

    def forward(self, x):
        # all about optimization with Lightning can be found here (e.g. how to define a particular optim step) :
        # https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        h = (self.width ** (-self.a[0])) * self.input_layer.forward(x)  # h_0 first layer pre-activations
        x = self.activation(h)  # x_0, first layer activations

        for l, layer in enumerate(self.intermediate_layers):  # L-1 intermediate layers
            h = (self.width ** (-self.a[l+1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = self.activation(h)  # x_l, l-th layer activations

        return (self.width ** (-self.a[self.n_layers-1])) * self.output_layer.forward(x)  # f(x)

    def predict(self, x, mode='probas', from_logits=False):
        if not from_logits:
            x = self.forward(x)
        if mode == 'probas':
            return torch.softmax(x, dim=-1)
        else:
            return x

    def training_step(self, batch, batch_nb):
        # If scheduler is used after every SGD step for the model, then the latter should override the optimizer_step
        # method of the lightning module.
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        with torch.no_grad():
            probas = self.predict(y_hat, mode='probas', from_logits=True)
            likelihood = probas[torch.arange(0, len(y)), y].mean()  # proba of correct label
            pred_proba, pred_label = torch.max(probas, dim=1)  # proba and index of predicted label
            acc = (pred_label == y).sum() / float(len(y))

        lrs_list = self._get_opt_lr()  # most abc_prams share the same lr for intermediate layers
        lrs = {'input_layer': lrs_list[0], 'intermediate_layers': lrs_list[1],
               'output_layer': lrs_list[-1]}

        tensorboard_logs = {'training/loss': loss, 'training/likelihood': likelihood, 'training/accuracy': acc,
                            'training/predicted_label_proba': pred_proba.mean()}
        for layer, lr in lrs.items():
            tensorboard_logs['learning_rates/{}'.format(layer)] = lr

        return {'output': y_hat.abs().mean(), 'loss': loss, 'likelihood': likelihood, 'accuracy': acc, 'pred_proba': pred_proba.mean(),
                'log': tensorboard_logs}

    # TODO : Apparently 'training_epoch_end' is not compatible with newer versions of PyTorch Lightning and might have
    #  to be changed to some other name such as 'on_epoch_end' for compliance with versions >= 0.9.0.
    def training_epoch_end(self, outputs):
        all_outputs = [x['output'] for x in outputs]
        all_losses = [x['loss'] for x in outputs]
        results = {'all_outputs': all_outputs,
                   'all_losses': all_losses,
                   'loss': np.mean(all_losses),
                   'likelihood': np.mean([x['likelihood'] for x in outputs]),
                   'accuracy': np.mean([x['accuracy'] for x in outputs]),
                   'pred_proba': np.mean([x['pred_proba'] for x in outputs]),
                   'lrs': [{key: x['log']['learning_rates/{}'.format(key)]
                            for key in ['input_layer', 'intermediate_layers', 'output_layer']}
                           for x in outputs]}

        self.results['training'].append(results)

        # lr decay if needed
        if (self.scheduler is not None) and (self.scheduler.lr_decay is not None):
            decay_factor = (self.scheduler.lr_decay ** self.current_epoch)
            new_lr = decay_factor * self.scheduler.base_lr
            logging.info("End of epoch {:,} going into epoch {:,}, new lr is {:.5f}".format(self.current_epoch,
                                                                                            self.current_epoch + 1,
                                                                                            new_lr))
            self.scheduler.set_param_group_lrs(new_lr)
        return results

    def _evaluation_step(self, batch, batch_nb, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        # torch.no_grad() should already be active here by construction of the Lightning module
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        probas = self.predict(y_hat, mode='probas', from_logits=True)
        likelihood = probas[torch.arange(0, len(y)), y].mean()  # proba of correct label
        pred_proba, pred_label = torch.max(probas, dim=1)  # proba and index of predicted label
        correct = pred_label == y
        acc = correct.sum() / float(len(y))

        return {'{}_loss'.format(short_name): loss, '{}_likelihood'.format(short_name): likelihood,
                '{}_accuracy'.format(short_name): acc, '{}_pred_proba'.format(short_name): pred_proba.mean(),
                'correct': correct}

    def _evaluation_epoch_end(self, outputs, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        avg_loss = torch.stack([x['{}_loss'.format(short_name)] for x in outputs]).mean()
        avg_likelihood = torch.stack([x['{}_likelihood'.format(short_name)] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['{}_accuracy'.format(short_name)] for x in outputs]).mean()

        # stack only works for tensors of the same size and the last batch might be of a different size
        correct = torch.cat([x['correct'] for x in outputs], dim=0)
        accuracy = correct.sum() / float(len(correct))

        tensorboard_logs = {'{}/loss'.format(mode): avg_loss, '{}/likelihood'.format(mode): avg_likelihood,
                            '{}/avg_accuracy'.format(mode): avg_accuracy, '{}/accuracy'.format(mode): accuracy}

        # append to validation/test results
        results = {'loss': avg_loss.item(), 'likelihood': avg_likelihood.item(), 'accuracy': accuracy.item(),
                   'log': tensorboard_logs}
        self.results[mode].append(results)

        return {'{}_loss'.format(short_name): avg_loss, '{}_likelihood'.format(short_name): avg_likelihood,
                '{}_accuracy'.format(short_name): accuracy, 'log': tensorboard_logs}
