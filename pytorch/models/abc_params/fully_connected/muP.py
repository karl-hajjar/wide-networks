import torch.nn as nn

from pytorch.models.abc_params.base_muP import BaseMuP


class FCmuP(BaseMuP):
    """
    A class implementing the NTK parameterization of a fully-connected network.
    """

    def __init__(self, config, width: int):
        """
        A class implementing abc-parameterizations for fully-connected networks.
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        super().__init__(config, width)

        if self._name == "model":
            self._name = "FCmuP"
        self._name = '{}_{}_{}'.format(self._name, self.width, self.n_layers)

        # define an attribute to hold all the history of train/val/test metrics for later plotting /analysing
        self.results = {'training': [], 'validation': [], 'test': []}

    def _build_model(self, config):
        """
        Simply build the necessary objects which will hold all the trainable parameters and name them appropriately.
        :param config: the configuration defining the loss, optimizer and architecture.
        :return:
        """
        # input layer
        self.input_layer = nn.Linear(in_features=config.architecture["input_size"],
                                     out_features=self.width,
                                     bias=True)

        # intermediate layers
        self.intermediate_layers = nn.ModuleList()
        for l in range(2, self.n_layers):
            self.intermediate_layers.add_module("layer_{:,}_intermediate".format(l),
                                                nn.Linear(in_features=self.width,
                                                          out_features=self.width,
                                                          bias=config.architecture["bias"]))

        # output layer
        self.output_layer = nn.Linear(in_features=self.width,
                                      out_features=config.architecture["output_size"],
                                      bias=config.architecture["bias"])

    def predict(self, x, mode='probas', from_logits=False):
        pass

    def training_step(self, batch, batch_nb):
        pass

    # TODO : Apparently 'training_epoch_end' is not compatible with newer versions of PyTorch Lightning and might have
    #  to be changed to some other name such as 'on_epoch_end' for compliance with versions >= 0.9.0.
    def training_epoch_end(self, outputs):
        pass

    def _evaluation_step(self, batch, batch_nb, mode: str):
        pass

    def _evaluation_epoch_end(self, outputs, mode: str):
        pass

    def configure_optimizers(self):
        pass
