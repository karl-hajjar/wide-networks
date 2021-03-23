import torch
from pytorch_lightning import LightningModule
import numpy as np

from pytorch.configs.model import ModelConfig
from pytorch.activations import get_activation
from pytorch.losses import get_loss
from pytorch.optimizers import get_optimizer
from pytorch.schedulers import get_scheduler
from pytorch.normalizations import get_norm
from pytorch.initializers import get_initializer


class BaseModel(LightningModule):
    """
    A base class implementing a generic neural network with the lightning module.
    """

    def __init__(self, config: ModelConfig):
        """
        Defines the model using the appropriate config class containing all the necessary parameters (activation, loss,
        loss, optimizer, training/eval batch size, ...)
        :param config: an object of the ModelConfig class.
        """
        super(BaseModel, self).__init__()
        self._check_config(config)
        self._name = config.name
        self._set_activation(config.activation)
        self._set_normalization(config.normalization)
        self._set_loss(config.loss)
        self._build_model(config)
        if len(list(self.parameters())) == 0:
            raise ValueError("Model has no parameters defined and optimizer cannot be defined: len(self.parameters) = "
                             "0. Parameters have to be defined in the _build_model() method.")
        else:  # only set the optimizer if some parameters have been already defined
            self._set_optimizer(config.optimizer)
        self._set_scheduler(config.scheduler)
        self.initialize_params(config.initializer)  # initialize the parameters using the config

        # define hparams for later logging
        self.hparams = config.dict()
        self.save_hyperparameters(config.dict())

        # define an attribute to hold all the history of train/val/test metrics for later plotting /analysing
        self.results = {'training': [], 'validation': [], 'test': []}

    def __str__(self):
        s = LightningModule.__str__(self)
        return self.name + " ({:,} params)".format(self.count_parameters()) + ":\n" + s

    @property
    def name(self):
        return self._name

    def _build_model(self, config):
        pass

    @staticmethod
    def _check_config(config):
        for attribute in ["name", "activation", "optimizer", "loss"]:
            if not hasattr(config, attribute):
                raise ValueError("config argument must have an attribute named '{}'".format(attribute))

    def _set_activation(self, activation_config):
        activation = get_activation(activation_config.name)
        if not hasattr(activation_config, "params"):
            self.activation = activation()
        else:
            try:
                self.activation = activation(**activation_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the activation : {}".format(e))

    def _set_loss(self, loss_config):
        loss = get_loss(loss_config.name)
        if not hasattr(loss_config, "params"):
            self.loss = loss()
        else:
            try:
                self.loss = loss(**loss_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the loss : {}".format(e))

    def _set_optimizer(self, optimizer_config, params=None):
        if params is None:
            params = self.parameters()
        optimizer = get_optimizer(optimizer_config.name)
        if not hasattr(optimizer_config, "params"):
            self.optimizer = optimizer(params)
        else:
            if optimizer_config.name == "adam":  # set betas = (beta1, beta2) in adam keyword args
                if ("beta1" in optimizer_config.params.keys()) and ("beta2" in optimizer_config.params.keys()):
                    beta1 = optimizer_config.params.pop("beta1")
                    beta2 = optimizer_config.params.pop("beta2")
                    optimizer_config.params["betas"] = (beta1, beta2)
            try:
                self.optimizer = optimizer(params, **optimizer_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the optimizer : {}".format(e))

    def _set_scheduler(self, scheduler_config=None):
        if scheduler_config is None:
            self.scheduler = None
        else:
            scheduler = get_scheduler(scheduler_config.name)
            if not hasattr(scheduler_config, "params"):
                self.scheduler = scheduler()
            else:
                try:
                    self.scheduler = scheduler(**scheduler_config.params)
                except Exception as e:
                    raise Exception("Exception while trying to create the scheduler : {}".format(e))

    def _set_normalization(self, norm_config):
        if norm_config is None:
            self.norm = None
        else:
            self.norm = get_norm(norm_config.name)

    def _set_initializer(self, init_config):
        initializer = get_initializer(init_config.name)
        if not hasattr(init_config, "params"):
            self.initializer = initializer(self.parameters())
        else:
            try:
                self.initializer = initializer(self.parameters(), **init_config.params)
            except Exception as e:
                raise Exception("Exception while trying to create the initializer : {}".format(e))

    def initialize_params(self, init_config=None):
        # initialise model parameters
        if init_config is None:
            initializer = get_initializer()
        else:
            initializer = get_initializer(init_config.name)
        for p in self.parameters():
            if p.dim() > 1:
                if not hasattr(init_config, "params"):
                    initializer(p)
                else:
                    try:
                        initializer(p, **init_config.params)
                    except Exception as e:
                        raise Exception("Exception while trying to initialize parameters : {}".format(e))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _get_opt_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        return np.mean(lrs)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'loss': avg_loss}
        return {'loss': avg_loss, 'log': logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        return self._evaluation_step(batch, batch_nb, mode="validation")

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        return self._evaluation_epoch_end(outputs, mode="validation")

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        return self._evaluation_step(batch, batch_nb, mode="test")

    def test_epoch_end(self, outputs):
        # OPTIONAL
        return self._evaluation_epoch_end(outputs, mode="test")

    def _evaluation_step(self, batch, batch_nb, mode: str):
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]
        x, y = batch
        y_hat = self(x)
        return {'{}_loss'.format(short_name): self.loss(y_hat, y)}

    def _evaluation_epoch_end(self, outputs, mode: str):
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]
        avg_loss = torch.stack([x['{}_loss'.format(short_name)] for x in outputs]).mean()
        logs = {'{}_loss'.format(short_name): avg_loss}
        return {'{}_loss'.format(short_name): avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return self.optimizer

    def train_dataloader(self):
        # OPTIONAL
        pass

    def val_dataloader(self):
        # OPTIONAL
        pass

    def test_dataloader(self):
        # OPTIONAL
        pass
