import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import List
import logging
from collections.abc import Iterable

from pytorch.models.base_model import BaseModel
from utils.nn import smoothed_exp_margin, smoothed_logistic_margin
from pytorch.initializers import INIT_DICT


class FCabcParam(BaseModel):
    """
    A class implementing an abc-parameterization of a fully-connected network. For more details on abc-parameterizations
    see https://arxiv.org/abs/2011.14522.
    """

    def __init__(self, config, width: int, a: List[float], b: List[float], c: [List[float], float]):
        """

        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        :param a: list of floats. The layer scales. The pre-activations of layer l will be scaled by m^{-a[l]}.
        :param b: list of floats. The initialization scales. The initialization of layer l will be scaled by m^{-b[l]}
        :param c: float or list of floats. The learning rate scales. The learning rate of layer l will be scaled by
        m^{-c[l]} if c is iterable, and by m^{-c} otherwise.
        """
        self._set_width(config.architecture, width)  # set width m and check that it is not None
        self._set_n_layers(config.architecture)  # set n_layers L+1 and check that it is not None
        self._set_bias(config.architecture)  # defines self.bias based on the config
        self._set_std(config.initializer)  # defines self.std based on the config

        self._check_scales(a, b, c)

        self.a = a  # layer scales
        self.b = b  # init scales
        if not (isinstance(c, Iterable)):
            c = [c] * self.n_layers
        elif len(c) == 1:
            c = [c[0]] * self.n_layers
        self.c = c  # learning rate scales

        # create optimizer, loss, activation, normalization
        super().__init__(config)
        if self._name == "model":
            self._name = "abcParam"
        self._name = '{}_{}_{}'.format(self._name, self.width, self.n_layers)

        # define an attribute to hold all the history of train/val/test metrics for later plotting /analysing
        self.results = {'training': [], 'validation': [], 'test': []}

    def _set_width(self, config, width):
        if config.architecture["width"] is not None:
            logging.info('Initializing width from config')
            self.width = config.architecture["width"]  # m
        elif width is not None:
            logging.info('Initializing width from argument')
            self.width = width  # m
        else:
            raise ValueError("Both the config and argument widths were None")

    def _set_n_layers(self, config):
        if config["n_layers"] is not None:
            self.n_layers = config["n_layers"]  # L + 1
        else:
            raise ValueError("`n_layers` was none in the config")

    def _set_bias(self, config):
        if ("bias" in config.keys()) and (config["bias"] is not None):
            self.bias = config["bias"]
        else:
            self.bias = False

    def _set_std(self, config):
        if ("std" in config["params"].keys()) and (config["params"]["std"] is not None):
            self.std = config["std"]
        else:
            self.std = 1.0

    def _check_scales(self, a, b, c):
        if len(a) != self.n_layers:
            raise ValueError("Layer scales `a` had {:,} elements but n_layers is {:,}")
        if len(b) != self.n_layers:
            raise ValueError("Init scales `b` had {:,} elements but n_layers is {:,}")
        if (isinstance(c, Iterable)) and (len(c) >= 2) and (len(c) != self.n_layers):
            raise ValueError("learning rate scales `c` had {:,} elements but n_layers is {:,}")

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
                                                          out_features=config.architecture["output_size"],
                                                          bias=config.architecture["bias"]))

        # output layer
        self.output_layer = nn.Linear(in_features=self.width,
                                      out_features=config.architecture["output_size"],
                                      bias=False)

    def forward(self, x):
        # all about optimization with Lightning can be found here (e.g. how to define a particular optim step) :
        # https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        h = (self.width ** (-self.a[0])) * self.input_layer.forward(x)  # h_0, first layer pre-activations
        x = self.activation(h)  # x_0, first layer activations

        for l, layer in enumerate(self.intermediate_layers):  # L-1 intermediate layers
            h = (self.width ** (-self.a[l+1])) * layer.forward(x)  # h_l, layer l pre-activations
            x = self.activation(h)  # x_l, l-th layer activations

        return (self.width ** (-self.a[self.n_layers-1])) * self.output_layer.forward(x)  # f(x)

    def initialize_params(self, init_config=None):
        """
        Initialize layer l matrix with m^{-b[l]} U^l where U^l_{ij} ~ N(0, 1) are iid (over i,j) Gaussians, and
        similarly layer l bias with m^{-b[l]} v^l where v^l_{i} ~ N(0, 1) are iid (over i) Gaussians.
        :param init_config:
        :return:
        """
        self._generate_standard_gaussians()
        with torch.no_grad():
            # weights
            self.input_layer.weight.data = (self.width ** (-self.b[0])) * self.U[0].data

            for l, layer in self.intermediate_layers:
                layer.weight.data = (self.width ** (-self.b[l+1])) * self.U[l+1].data

            self.output_layer.weight.data = (self.width ** (-self.b[self.n_layers-1])) * self.U[self.n_layers-1].data

            # biases
            if hasattr(self.input_layer, "bias"):
                self.input_layer.bias.data = (self.width ** (-self.b[0])) * self.v[0].data

            if self.bias:
                for l, layer in self.intermediate_layers:
                    layer.bias.data = (self.width ** (-self.b[l + 1])) * self.v[l + 1].data

                self.output_layer.bias.data = (self.width ** (-self.b[self.n_layers-1])) * self.v[self.n_layers-1].data

    def _generate_standard_gaussians(self, std=2.0):
        self.U = [torch.normal(mean=0, std=self.std, size=self.input_layer.weight.size(), requires_grad=False)]
        self.U += [torch.normal(mean=0, std=self.std, size=(self.width, self.width), requires_grad=False)
                   for _ in self.intermediate_layers]
        self.U.append(torch.normal(mean=0, std=self.std, size=self.output_layer.weight.size(), requires_grad=False))

        if hasattr(self.input_layer, "bias"):
            self.v = [torch.normal(mean=0, std=self.std, size=self.input_layer.bias.size(), requires_grad=False)]

        if self.bias:
            self.v += [torch.normal(mean=0, std=self.std, size=layer.bias.size(), requires_grad=False)
                       for layer in self.intermediate_layers]
            self.v.append(torch.normal(mean=0, std=self.std, size=self.output_layer.bias.size(), requires_grad=False))

    def predict(self, x, mode='probas', from_logits=False):
        if not from_logits:
            x = self.forward(x)
        if mode == 'probas':
            return torch.sigmoid(x)
        else:
            return x

    @property
    def beta(self):
        # For 'neurons' parameterized by w = (a,b), the scaled squared-norm (1/m) * sum_i ||w_i||^2 is equal to :
        # (1/m) * sum_i (a_i^2 + ||b_i||^2) = (1/m) * ||a||^2 + ||[b1; ...; bm]||_F^2
        # The computation of beta is done according to https://github.com/lchizat/2020-implicit-bias-wide-2NN/blob/3acfbb441cf8b235a9982497553a4c25d9ee6623/implicit_bias_2NN_utils.jl#L36
        with torch.no_grad():  # remove gradient computation when computing the norm
            norm = (self.layer2.weight.detach() ** 2).sum() + (self.layer1.weight.detach() ** 2).sum()
            if hasattr(self.layer1, "bias"):
                norm += (self.layer1.bias.detach() ** 2).sum()
            beta = (1 / self.hidden_layer_dim) * norm
        return beta

    def train_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def val_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def test_dataloader(self, dataset=None, shuffle=True, batch_size=32, indexes=None):
        dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_0_1 = (y + 1) / 2  # converting target labels from {-1, 1} to {0, 1}.
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y_0_1)

        sample_margins = y * y_hat  # y assumed to have values in {-1, 1}
        likelihood = self.predict(sample_margins, mode='probas', from_logits=True).mean()
        probas = self.predict(y_hat, mode='probas', from_logits=True)
        pred_label = (probas >= 0.5).long()
        pred_proba = torch.max(probas, 1 - probas)  # element-wise max of the 2 tensors
        acc = (pred_label == y_0_1).sum() / float(len(y))

        # margin and weights norm
        margin = sample_margins.min()
        beta = self.beta
        # normalized margin should be bounded by max(1, r^2)/2 where r is a bound on the norm of the data points
        normalized_margin = margin / beta  # see https://github.com/lchizat/2020-implicit-bias-wide-2NN/blob/3acfbb441cf8b235a9982497553a4c25d9ee6623/implicit_bias_2NN_utils.jl#L38

        smoothed_exp_m = smoothed_exp_margin(beta, sample_margins)
        smoothed_log_m = smoothed_logistic_margin(beta, sample_margins)

        # get probability of target labels only and then average over batch
        tensorboard_logs = {'training/loss': loss, 'training/likelihood': likelihood, 'training/accuracy': acc,
                            'training/predicted_label_proba': pred_proba.mean(), 'training/margin': margin,
                            'training/normalized_margin': normalized_margin, 'training/beta': beta,
                            'training/smoothed_exp_margin': smoothed_exp_m,
                            'training/smoothed_logistic_margin': smoothed_log_m,
                            'training/learning_rate': self._get_opt_lr()}

        return {'loss': loss, 'likelihood': likelihood, 'accuracy': acc, 'margin': margin,
                'normalized_margin': normalized_margin, 'sample_margins': sample_margins, 'log': tensorboard_logs}

    # TODO : Apparently 'training_epoch_end' is not compatible with newer versions of PyTorch Lightning and might have
    #  to be changed to some other name such as 'on_epoch_end' for compliance with versions >= 0.9.0.
    def training_epoch_end(self, outputs):
        beta = self.beta
        margin = np.min([x['margin'] for x in outputs])
        sample_margins = torch.cat([x['sample_margins'] for x in outputs], dim=0)
        smoothed_exp_m = smoothed_exp_margin(beta, sample_margins)
        smoothed_log_m = smoothed_logistic_margin(beta, sample_margins)
        results = {'loss': np.mean([x['loss'] for x in outputs]),
                   'likelihood': np.mean([x['likelihood'] for x in outputs]),
                   'accuracy': np.mean([x['accuracy'] for x in outputs]),
                   'margin': margin,
                   'normalized_margin': margin / beta,
                   'smoothed_exp_margin': smoothed_exp_m,
                   'smoothed_logistic_margin': smoothed_log_m,
                   'beta': beta}
        self.results['training'].append(results)
        # optionally, results could instead be flushed for training and validation at the end of every epoch
        return results

    def _evaluation_step(self, batch, batch_nb, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        x, y = batch
        y_hat = self.forward(x)
        y_0_1 = (y + 1) / 2  # converting target labels from {-1, 1} to {0, 1}.

        sample_margins = y * y_hat  # y assumed to have values in {-1, 1}
        likelihood = self.predict(sample_margins, mode='probas', from_logits=True).mean()
        probas = self.predict(y_hat, mode='probas', from_logits=True)
        pred_label = (probas >= 0.5).long()
        pred_proba = torch.max(probas, 1 - probas)  # element-wise max of the 2 tensors
        correct = pred_label == y_0_1
        acc = correct.sum() / float(len(y))

        # margin and weights norm
        margin = sample_margins.min()

        return {'{}_loss'.format(short_name): self.loss(y_hat, y_0_1), '{}_likelihood'.format(short_name): likelihood,
                '{}_accuracy'.format(short_name): acc, '{}_pred_proba'.format(short_name): pred_proba.mean(),
                '{}_margin'.format(short_name): margin, '{}_sample_margins'.format(short_name): sample_margins,
                'pred_probas': pred_proba, 'correct': correct}

    def _evaluation_epoch_end(self, outputs, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        avg_loss = torch.stack([x['{}_loss'.format(short_name)] for x in outputs]).mean()
        avg_likelihood = torch.stack([x['{}_likelihood'.format(short_name)] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['{}_accuracy'.format(short_name)] for x in outputs]).mean()
        avg_pred_proba = torch.stack([x['{}_pred_proba'.format(short_name)] for x in outputs]).mean()

        sample_margins = torch.cat([x['{}_sample_margins'.format(short_name)] for x in outputs], dim=0)
        margin = torch.stack(([x['{}_margin'.format(short_name)] for x in outputs])).min()
        beta = self.beta

        normalized_margin = margin / beta
        smoothed_exp_m = smoothed_exp_margin(beta, sample_margins)
        smoothed_log_m = smoothed_logistic_margin(beta, sample_margins)

        # stack only works for tensors of the same size and the last batch might be of a different size
        pred_probas = torch.cat([x['pred_probas'] for x in outputs], dim=0)
        correct = torch.cat([x['correct'] for x in outputs], dim=0)
        y_true = correct.detach().cpu().numpy()
        if (y_true >= 1.0).all():
            auc = 1.0
        else:
            auc = roc_auc_score(y_true=y_true, y_score=pred_probas.detach().cpu().numpy())

        tensorboard_logs = {'{}/loss'.format(mode): avg_loss, '{}/likelihood'.format(mode): avg_likelihood,
                            '{}/accuracy'.format(mode): avg_accuracy, '{}/roc_auc'.format(mode): auc,
                            '{}/predicted_label_proba'.format(mode): avg_pred_proba, '{}/margin'.format(mode): margin,
                            '{}/normalized_margin'.format(mode): normalized_margin,
                            '{}/smoothed_exp_margin'.format(mode): smoothed_exp_m,
                            '{}/smoothed_logistic_margin'.format(mode): smoothed_log_m}

        # append to validation results
        results = {'loss': avg_loss, 'likelihood': avg_likelihood, 'accuracy': avg_accuracy, 'margin': margin,
                   'normalized_margin': normalized_margin, 'smoothed_exp_margin': smoothed_exp_m,
                   'smoothed_logistic_margin': smoothed_log_m}
        self.results[mode].append(results)
        # optionally, results could instead be flushed for training and validation at the end of every epoch

        return {'{}_loss'.format(short_name): avg_loss, '{}_likelihood'.format(short_name): avg_likelihood,
                '{}_accuracy'.format(short_name): avg_accuracy, '{}_auc'.format(short_name): auc,
                '{}_pred_proba'.format(short_name): avg_pred_proba, '{}_margin'.format(short_name): margin,
                '{}_normalized_margin'.format(short_name): normalized_margin, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # TODO : add schedulers.py for handling different types of schedulers and add those to the config class and file
        # with this scheduler, learning rate will be updated at the END of each epoch only
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda i: 1 / np.sqrt(i + 1))
        return [self.optimizer], [scheduler]
