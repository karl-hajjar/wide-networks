from typing import List

from .base_abc_param import BaseABCParam


class BaseIP(BaseABCParam):
    """
    A base class implementing the general skeleton of the IP parameterization which corresponds to the parameterization
    often used in Mean Field models of NNs. Anything that is architecture specific is left out of this class and has to
    be implemented in the child classes.
    """

    def __init__(self, config,  c: [List[float], float], width: int = None):
        """
        Base class for the IP parameterization with L hidden layers where:
         - a[0] = 0, a[l] = 1 for l in [1, L]
         - b[l] = 0, for any l in [0, L]
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers - 1
        a = [0] + [1 for _ in range(1, L+1)]
        b = [0 for _ in range(L+1)]

        super().__init__(config, a, b, c, width)
