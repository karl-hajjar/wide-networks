from .base_ip import BaseIP
from pytorch.models.base_model import BaseModel
from pytorch.schedulers import WarmupSwitchLRBias
from utils.nn import get_standard_mf_lr_exponents

import torch
import math


class BaseIPLLRBias(BaseIP, BaseModel):
    """
    A base class implementing the general skeleton of IP parameterization with Large Learning Rates (IP-LLR). This is a
    version of the IP-parameterization where the initial learning rates are made large to allow training to start at
    depths >= 4 with large widths. Anything that is architecture specific is left out of this class and has to be
    implemented in the child classes.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1, lr_calibration_batches: list = None):
        """
        Base class for the IP-LLR parameterization with biases that are not scaled with the width. The initial learning
        rates are large and then switched to the the learning rates of standard Mean Field models after a certain number
        n_warmup_steps of optimization steps:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
        For the weights:
         - c[0] = -(L+1) / 2, c[l] = -(L - (l+1) + 4) / 2 for l in [1, L-1], c[L] = -1 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        For the biases:
         - c[l] = -(L - (l+1) + 2) / 2 for l in [0, L-1], c[L] = 0 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before switching
        to the standard Mean Field learning rates.
        """
        self._set_n_layers(config.architecture)
        self._set_scale_bias_lr(config.architecture)
        L = self.n_layers - 1
        # set c to the initial learning rates exponents
        c = [-(L+1) / 2] + [-(L - l + 4) / 2 for l in range(2, L+1)] + [-1]

        self._set_n_warmup_steps(config.scheduler, n_warmup_steps)
        warm_lr_exponents = get_standard_mf_lr_exponents(L)
        self.width = config.architecture['width']

        if self.scale_bias_lr:
            bias_lr_exponents = [-(L - l + 2) / 2 for l in range(1, L+1)] + [0]
            self.bias_lr_scales = self._set_scales_from_exponents(bias_lr_exponents)
        else:
            self.bias_lr_scales = [1.0 for l in range(1, L+2)]

        self.warm_lrs = self._set_scales_from_exponents(warm_lr_exponents)
        if self.scale_bias_lr:
            self.warm_bias_lrs = self._set_scales_from_exponents([-1 for l in range(1, L+1)] + [0])
        else:
            self.warm_bias_lrs = [1.0 for l in range(1, L+2)]

        # only used when defining the scheduler if the latter automatically calibrates the initial lr
        self.lr_calibration_batches = lr_calibration_batches

        # reset bias parameters in the config
        config.architecture["bias"] = True
        config.architecture["scale_bias"] = False
        BaseIP.__init__(self, config, c, width)

    def _set_scale_bias_lr(self, config):
        if 'scale_bias_lr' in config.keys():
            self.scale_bias_lr = config["scale_bias_lr"]
        else:
            self.scale_bias_lr = False

    def _set_n_warmup_steps(self, scheduler_config, n_warmup_steps):
        """
        Sets the attribute `n_warmup_steps` from the scheduler config if it appears there, otherwise from the argument.
        :param scheduler_config:
        :return:
        """
        if (hasattr(scheduler_config, 'params')) and ('n_warmup_steps' in scheduler_config.params.keys()):
            self.n_warmup_steps = scheduler_config.params['n_warmup_steps']
        else:
            self.n_warmup_steps = n_warmup_steps

    def _set_optimizer(self, optimizer_config, params=None):
        if (optimizer_config.params is None) or ('lr' not in optimizer_config.params.keys()):
            raise ValueError("optimizer config must have a parameter `lr` in abc-parameterizations")
        else:
            self.base_lr = optimizer_config.params['lr']

        # Layer-wise learning rates : one parameter group for the weights and one for the bias of each layer with their
        # own lr
        weight_param_groups = \
            [{'params': self.input_layer.weight, 'lr': self.base_lr * self.lr_scales[0], 'name': 'weights_1'}] + \
            [{'params': layer.weight, 'lr': self.base_lr * self.lr_scales[l+1], 'name': 'weights_{}'.format(l+2)}
             for l, layer in enumerate(self.intermediate_layers)] + \
            [{'params': self.output_layer.weight, 'lr': self.base_lr * self.lr_scales[self.n_layers - 1],
              'name': 'weights_{}'.format(self.n_layers)}]

        bias_param_groups = \
            [{'params': self.input_layer.bias, 'lr': self.base_lr * self.bias_lr_scales[0], 'name': 'bias_1'}] + \
            [{'params': layer.bias, 'lr': self.base_lr * self.bias_lr_scales[l+1], 'name': 'bias_{}'.format(l+2)}
             for l, layer in enumerate(self.intermediate_layers)] + \
            [{'params': self.output_layer.bias, 'lr': self.base_lr * self.bias_lr_scales[self.n_layers - 1],
              'name': 'bias_{}'.format(self.n_layers)}]

        param_groups = weight_param_groups + bias_param_groups

        BaseModel._set_optimizer(self, optimizer_config, params=param_groups)

    def _set_scheduler(self, scheduler_config=None):
        if not hasattr(scheduler_config, "params"):
            self.scheduler = WarmupSwitchLRBias(self.optimizer, initial_lrs=self.lr_scales, warm_lrs=self.warm_lrs,
                                                initial_bias_lrs=self.bias_lr_scales, warm_bias_lrs=self.warm_bias_lrs,
                                                base_lr=self.base_lr, model=self, batches=self.lr_calibration_batches)
        else:
            try:
                self.scheduler = WarmupSwitchLRBias(self.optimizer, initial_lrs=self.lr_scales, warm_lrs=self.warm_lrs,
                                                    initial_bias_lrs=self.bias_lr_scales,
                                                    warm_bias_lrs=self.warm_bias_lrs,
                                                    base_lr=self.base_lr, model=self,
                                                    batches=self.lr_calibration_batches,
                                                    **scheduler_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the scheduler : {}".format(e))

    def forward(self, x):
        h = self.input_layer.forward((self.width ** (-self.a[0])) * x)  # h_0 first layer pre-activations
        x = self.activation(h)  # x_0, first layer activations

        for l, layer in enumerate(self.intermediate_layers):  # L-1 intermediate layers
            # by putting the scaling factor in the input x, we prevent the bias from being scaled as well, which is the
            # desired effect
            h = layer.forward((self.width ** (-self.a[l+1])) * x)  # h_l, layer l pre-activations
            x = self.activation(h)  # x_l, l-th layer activations

        return self.output_layer.forward((self.width ** (-self.a[self.n_layers-1])) * x)  # f(x)

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int, second_order_closure=None,
                       on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False):
        """
        Override base method `optimizer_step` from pytorch.Lightning to add scheduler step after every optimization step
        (and not simply at every epoch).
        :return:
        """
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, on_tpu,
                               using_native_amp, using_lbfgs)
        self.scheduler.step()  # take scheduler step right after optimization step