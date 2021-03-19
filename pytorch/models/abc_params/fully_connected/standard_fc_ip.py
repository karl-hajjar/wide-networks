from torch import nn

from pytorch.models.abc_params.standard_ip import StandardIP
from .ip import FCIP


class StandardFCIP(StandardIP, FCIP):
    """
    A class implementing a fully-connected IP parameterization with the standard learning rates found in the Mean Field
    literature.
    """

    def __init__(self, config, width: int = None):
        """
        Class for a fully-connected IP parameterization with the standard learning rates of Mean Field models:
         - a[0] = 0, a[l] = 1 for l in [1, L-1]
         - b[l] = 0, for any l in [0, L-1]
         - c[0] = -1, c[l] = -2 for l in [1, L-2], c[L-1] = -1
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """

        StandardIP.__init__(self, config, width)

    def _build_model(self, config):
        FCIP._build_model(self, config)

