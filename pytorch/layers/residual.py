from pytorch_lightning import LightningModule


class ResidualConnection(LightningModule):
    """
    A class defining the residual connection.
    """

    def __init__(self, activation):
        super(ResidualConnection, self).__init__()
        self.activation = activation

    def forward(self, x, module):
        return self.activation(x + module(x))
