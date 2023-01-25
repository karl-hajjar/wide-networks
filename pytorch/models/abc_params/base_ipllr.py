from .base_ip import BaseIP
from pytorch.schedulers import WarmupSwitchLR
from utils.nn import get_standard_mf_lr_exponents


class BaseIPLLR(BaseIP):
    """
    A base class implementing the general skeleton of IP parameterization with Large Learning Rates (IP-LLR). This is a
    version of the IP-parameterization where the initial learning rates are made large to allow training to start at
    depths >= 4 with large widths. Anything that is architecture specific is left out of this class and has to be
    implemented in the child classes.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1, lr_calibration_batches: list = None):
        """
        Base class for the IP-LLR parameterization with L hidden layers where the initial learning rates are large and
        then switched to the learning rates of standard Mean Field models after a certain number n_warmup_steps of
        optimization steps:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
         - c[0] = -(L+1) / 2, c[l] = -(L+2) / 2 for l in [1, L-1], c[L] = -(L+1) / 2 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before switching
        to the standard Mean Field learning rates.
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers - 1
        # set c to the initial learning rates exponents
        c = [-(L+1) / 2] + [-(L+2) / 2 for _ in range(1, L)] + [-(L+1) / 2]

        self._set_n_warmup_steps(config.scheduler, n_warmup_steps)
        warm_lr_exponents = get_standard_mf_lr_exponents(L)
        self.width = config.architecture['width']
        self.warm_lrs = self._set_scales_from_exponents(warm_lr_exponents)

        # only used when defining the scheduler if the latter automatically calibrates the initial lr
        self.lr_calibration_batches = lr_calibration_batches

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

    # TODO : if bias is set to True, then the learning rates for the biases need to be set appropriately and differently
    #  from the weights: for IP-LLR they need to scale as dh in m^{-1}m^{(L-l)/2}.
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
