from pytorch.models.abc_params.base_ipllr import BaseIPLLR
from .ip import FCIP


class FcIPLLR(BaseIPLLR, FCIP):
    """
    A class implementing a fully-connected IP-LLR parameterization with large initial learning rates.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1):
        """
        Class for a fully-connected IP-LLR parameterization with large initial learning rates:
          - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
         - c[0] = -(L+1) / 2, c[l] = -(L+2) / 2 for l in [1, L-1], c[L] = -(L+1) / 2 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before switching
        to the standard Mean Field learning rates.
        """
        BaseIPLLR.__init__(self, config, width, n_warmup_steps)

    def _build_model(self, config):
        FCIP._build_model(self, config)

