import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np

from .base_model import BaseModel
from utils.nn import smoothed_exp_margin, smoothed_logistic_margin
from pytorch.initializers import INIT_DICT


class TwoLayerNet(BaseModel):
    """
    A class implementing a 2 layer fully connected neural network to be trained on a binary classification task.
    """

    def __init__(self, config, train_hidden=True):
        self.hidden_layer_dim = config.architecture["hidden_layer_dim"]
        # TODO : check how to freeze layers in PyTorch
        self.train_hidden = train_hidden

        # create optimizer, loss, activation, normalization
        super(TwoLayerNet, self).__init__(config)
        if self._name == "model":
            self._name = "TwoLayerNet"
        self._name = self._name + str(self.hidden_layer_dim)

        # define an attribute to hold all the history of train/val/test metrics for later plotting /analysing
        self.results = {'training': [], 'validation': [], 'test': []}

    def _build_model(self, config):
        # define both layers
        self.layer1 = nn.Linear(in_features=config.architecture["input_size"],
                                out_features=self.hidden_layer_dim,
                                bias=True)
        self.layer2 = nn.Linear(in_features=self.hidden_layer_dim,
                                out_features=1,
                                bias=False)

    def initialize_params(self, init_config=None):
        """
        Initializes by default the hidden layer with uniform distribution on the sphere and output weights uniformly in
        {-1, 1}. If an `init_config' is provided with a valid name (e.g. 'glorot_uniform') then the corresponding
        initialization is used.
        :param init_config:
        :return:
        """
        if (init_config is None) or (init_config.name not in INIT_DICT.keys()):
            if (init_config is None) or (not hasattr(init_config, 'params')) or \
               ('std' not in init_config.params.keys()):
                std = 1.0
            else:
                std = init_config['params']['std']
            with torch.no_grad():  # disable auto-grad for initialization
                (m, d) = self.layer1.weight.shape
                if hasattr(self.layer1, "bias"):
                    for j in range(m):  # handle 'neurons' one by one
                        u = torch.randn(d+1)
                        norm = ((u ** 2).sum()) ** 0.5
                        self.layer1.weight[j, :] = std * u[:d] / (norm + 1e-6)  # j-th weight of hidden layer
                        self.layer1.bias[j] = u[d] / (norm + 1e-6)  # j-th bias of hidden layer
                        if torch.rand(1).item() >= 0.5:  # handle j-th entry of output layer
                            self.layer2.weight[0, j] = 1.0
                        else:
                            self.layer2.weight[0, j] = -1.0
                else:
                    for j in range(m):  # handle 'neurons' one by one
                        u = torch.randn(d)
                        self.layer1.weight[j, :] = std * u / (((u ** 2).sum()) ** 0.5 + 1e-6)  # j-th weight
                        if torch.rand(1).item() >= 0.5:  # handle j-th entry of output layer
                            self.layer2.weight[0, j] = 1.0
                        else:
                            self.layer2.weight[0, j] = -1.0
        else:
            super(TwoLayerNet, self).initialize_params(init_config)

    def forward(self, x):
        # all about optimization with Lightning can be found here (e.g. how to define a particular optim step) :
        # https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        return (1 / self.hidden_layer_dim) * self.layer2(self.activation(self.layer1(x)))

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
