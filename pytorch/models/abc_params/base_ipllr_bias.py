from .base_ip import BaseIP
from pytorch.schedulers import WarmupSwitchLR
from utils.nn import get_standard_mf_lr_exponents

import torch
import math


class BaseIPLLRBias(BaseIP):
    """
    A base class implementing the general skeleton of IP parameterization with Large Learning Rates (IP-LLR). This is a
    version of the IP-parameterization where the initial learning rates are made large to allow training to start at
    depths >= 4 with large widths. Anything that is architecture specific is left out of this class and has to be
    implemented in the child classes.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1, lr_calibration_batches: list = None):
        """
        Base class for the IP-LLR parameterization with biases that are not scaled with the width. Tthe initial learning
        rates are large and then switched to the the learning rates of standard Mean Field models after a certain number
        n_warmup_steps of optimization steps:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
         - c[0] = -(L+1) / 2, c[l] = -(L - l + 3) / 2 for l in [1, L-1], c[L] = -1 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before switching
        to the standard Mean Field learning rates.
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers - 1
        # set c to the initial learning rates exponents
        c = [-(L+1) / 2] + [-(L - l + 3) / 2 for l in range(1, L)] + [-1]

        self._set_n_warmup_steps(config.scheduler, n_warmup_steps)
        self.n_warmup_steps = n_warmup_steps
        warm_lr_exponents = get_standard_mf_lr_exponents(L)
        self.width = config.architecture['width']
        self.warm_lrs = self._set_scales_from_exponents(warm_lr_exponents)

        # only used when defining the scheduler if the latter automatically calibrates the initial lr
        self.lr_calibration_batches = lr_calibration_batches

        # reset bias parameters in the config
        config.architecture["bias"] = True
        config.architecture["scale_bias"] = False
        super().__init__(config, c, width)

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

    def _set_scheduler(self, scheduler_config=None):
        if not hasattr(scheduler_config, "params"):
            self.scheduler = WarmupSwitchLR(self.optimizer, initial_lrs=self.lr_scales, warm_lrs=self.warm_lrs,
                                            base_lr=self.base_lr, model=self, batches=self.lr_calibration_batches)
        else:
            try:
                self.scheduler = WarmupSwitchLR(self.optimizer, initial_lrs=self.lr_scales, warm_lrs=self.warm_lrs,
                                                base_lr=self.base_lr, model=self, batches=self.lr_calibration_batches,
                                                **scheduler_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the scheduler : {}".format(e))

    def forward(self, x, normalize_first=True):
        h = (self.width ** (-self.a[0])) * self.input_layer.forward(x)  # h_0 first layer pre-activations
        if normalize_first:
            h = h / math.sqrt(self.d + 1)
        x = self.activation(h)  # x_0, first layer activations

        for l, layer in enumerate(self.intermediate_layers):  # L-1 intermediate layers
            # by putting the scaling factor in the input x, we prevent the bias from being scaled as well, which is the
            # desired effect
            h = layer.forward((self.width ** (-self.a[l+1])) * x)  # h_l, layer l pre-activations
            x = self.activation(h)  # x_l, l-th layer activations

        return self.output_layer.forward((self.width ** (-self.a[self.n_layers-1])) * x)  # f(x)

    def training_step(self, batch, batch_nb):
        out = super().training_step(batch, batch_nb)
        self.scheduler.step()
        return out
