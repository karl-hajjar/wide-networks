from pytorch.models.abc_params.base_muP import BaseMuP
from .ip import FCIP


class FCmuP(BaseMuP, FCIP):
    """
    A class implementing the NTK parameterization of a fully-connected network.
    """

    def __init__(self, config, width: int = None):
        """
        A class implementing abc-parameterizations for fully-connected networks.
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        BaseMuP.__init__(self, config, width)

        if self._name == "model":
            self._name = "FCmuP"
        self._name = '{}_{}_{}'.format(self._name, self.width, self.n_layers)

    def _build_model(self, config):
        """
        Simply build the necessary objects which will hold all the trainable parameters and name them appropriately.
        :param config: the configuration defining the loss, optimizer and architecture.
        :return:
        """
        FCIP._build_model(self, config)
