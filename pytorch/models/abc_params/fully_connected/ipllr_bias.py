from pytorch.models.abc_params.base_ipllr_bias import BaseIPLLRBias
from .ip import FCIP


class FcIPLLRBias(BaseIPLLRBias, FCIP):
    """
    A class implementing a fully-connected IP-LLR parameterization with large initial learning rates and biases that are
    not scaled with the width m.
    """

    def __init__(self, config, width: int = None, n_warmup_steps: int = 1, lr_calibration_batches: list = None):
        """
        Class for a fully-connected IP-LLR parameterization with large initial learning rates:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
         - c[0] = -(L+1) / 2, c[l] = -(L - l + 3) / 2 for l in [1, L-1], c[L] = -1 if t < n_warmup_steps
         - c[0] = -1, c[l] = -2 for l in [1, L-1], c[L] = -1 if t >= n_warmup_steps
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param n_warmup_steps: the number of optimization steps to take with the initial learning rates before switching
        to the standard Mean Field learning rates.
        """
        BaseIPLLRBias.__init__(self, config, width, n_warmup_steps, lr_calibration_batches)

    def _build_model(self, config):
        FCIP._build_model(self, config)


