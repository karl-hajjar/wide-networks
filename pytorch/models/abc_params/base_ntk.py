from .base_abc_param import BaseABCParam


class BaseNTK(BaseABCParam):
    """
    A base class implementing the general skeleton of the NTK parameterization. For more details on how NTK is expressed
    as an abc-parameterization see https://arxiv.org/abs/2011.14522. Anything that is architecture specific is left out
    of this class and has to be implemented in the child classes.
    """

    def __init__(self, config, width: int = None, results_path=None):
        """
        Base class for the NTK parameterization where:
         - a[0] = 0, a[l] = 1/2 for any l in [1, L-1]
         - b[l] = 0, for any l in [0, L-1]
         - c = 0
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers
        a = [0.] + [1/2 for _ in range(1, L)]
        b = [0 for _ in range(L)]
        c = 0

        # create optimizer, loss, activation, normalization
        super().__init__(config, a, b, c, width, results_path)
