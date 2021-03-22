from .base_ip import BaseIP


class BaseIPLLR(BaseIP):
    """
    A base class implementing the general skeleton of IP parameterization with Large Learning Rates (IP-LLR). This is a
    version of the IP-parameterization where the initial learning rates are made large to allow training to start at
    depths >= 4 with large widths. Anything that is architecture specific is left out of this class and has to be
    implemented in the child classes.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1):
        """
        Base class for the IP-LLR parameterization where the initial learning rates are large and then switched to the
        the learning rates of standard Mean Field models after a certain number n_warmup_steps of optimization steps:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
         - c[0] = -(L+1) / 2, c[l] = -(L+2) / 2 for l in [1, L-1], c[L] = -(L+1) / 2 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before s
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers
        # set c to the initial learning rates values
        c = [-(L+1) / 2] + [-(L+2) / 2 for _ in range(1, L-1)] + [-(L+1) / 2]
        self.n_warmup_steps = n_warmup_steps

        super().__init__(config, c, width)
