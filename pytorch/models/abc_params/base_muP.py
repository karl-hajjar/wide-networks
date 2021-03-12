from .base_abc_param import BaseABCParam


class BaseMuP(BaseABCParam):
    """
    A base class implementing the general skeleton of the muP parameterization. For more details on the muP
    parameterization see https://arxiv.org/abs/2011.14522. Anything that is architecture specific is left out of this
    class and has to be implemented in the child classes.
    """

    def __init__(self, config, width: int):
        """
        Base class for the muP parameterization where:
         - a[0] = -1/2, a[l] = 0 for l in [1,L-2], a[L-1] = 1/2
         - b[l] = 1/2, for any l in [0, L-1]
         - c = 0
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        self._set_n_layers(config.architecture)
        L = self.n_layers
        a = [-1/2] + [0 for _ in range(1, L-1)] + [1/2]
        b = [1/2 for _ in range(L)]
        c = 0

        # create optimizer, loss, activation, normalization
        super().__init__(config, a, b, c, width)
