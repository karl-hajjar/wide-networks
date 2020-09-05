import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np

from .resnet import ResNet


class ResNetMNIST(ResNet):
    """
    A class implementing a resnet CNN architecture for the digit classification problem on the MNIST dataset.
    """

    softmax = nn.Softmax(dim=1)

    def __init__(self, config):
        super(ResNetMNIST, self).__init__(config)

    def predict(self, x, mode='probas', from_logits=False):
        if not from_logits:
            x = self.forward(x)
        if mode == 'probas':
            return self.softmax(x)
        else:
            return x

    def get_likelihood(self, x: torch.Tensor, y: torch.Tensor, from_logits=False, reduce='mean'):
        all_probas = self.predict(x, mode='probas', from_logits=from_logits)
        target_probas = all_probas[:, y]
        if reduce == 'mean':
            return target_probas.mean()
        elif reduce == 'sum':
            return target_probas.mean()
        else:
            return target_probas

    def train_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = MNIST(data_dir, train=True, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def val_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = MNIST(data_dir, train=True, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def test_dataloader(self, data_dir=None, download=False, shuffle=True, batch_size=32, indexes=None):
        if data_dir is None:
            data_dir = os.getcwd()
        dataset = MNIST(data_dir, train=False, download=download, transform=transforms.ToTensor())
        if indexes is not None:
            dataset = Subset(dataset, indexes)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        all_probas = self.predict(y_hat, mode='probas', from_logits=True)
        pred_proba, pred_label = torch.max(all_probas, 1)
        acc = (pred_label == y).sum() / float(len(y))
        # get probability of target labels only and then average over batch
        likelihood = all_probas[torch.arange(0, len(y), dtype=torch.long), y].mean()
        tensorboard_logs = {'training/loss': loss, 'training/likelihood': likelihood, 'training/accuracy': acc,
                            'training/predicted_label_proba': pred_proba.mean(),
                            'training/learning_rate': self._get_opt_lr()}
        # to add values to be logged in the progress bar include a key {'progress_bar': logs} in the returned dict
        # return {'loss': loss, 'likelihood': likelihood, 'accuracy': acc, 'log': tensorboard_logs}
        return {'loss': loss, 'likelihood': likelihood, 'accuracy': acc, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def _evaluation_step(self, batch, batch_nb, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        x, y = batch
        y_hat = self(x)
        all_probas = self.predict(y_hat, mode='probas', from_logits=True)
        pred_proba, pred_label = torch.max(all_probas, 1)
        correct = pred_label == y
        acc = correct.sum() / float(len(y))
        # get probability of target labels only and then average over batch
        likelihood = all_probas[torch.arange(0, len(y), dtype=torch.long), y].mean()
        return {'{}_loss'.format(short_name): self.loss(y_hat, y), '{}_likelihood'.format(short_name): likelihood,
                '{}_accuracy'.format(short_name): acc, '{}_pred_proba'.format(short_name): pred_proba.mean(),
                'pred_probas': pred_proba, 'correct': correct}

    def _evaluation_epoch_end(self, outputs, mode: str):
        # define prefixes
        mode_to_short = {'validation': 'val', 'test': 'test'}
        short_name = mode_to_short[mode]

        avg_loss = torch.stack([x['{}_loss'.format(short_name)] for x in outputs]).mean()
        avg_likelihood = torch.stack([x['{}_likelihood'.format(short_name)] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['{}_accuracy'.format(short_name)] for x in outputs]).mean()
        avg_pred_proba = torch.stack([x['{}_pred_proba'.format(short_name)] for x in outputs]).mean()
        # stack only works for tensors of the same size and the last batch might be of a different size
        pred_probas = torch.cat([x['pred_probas'] for x in outputs], dim=0)
        correct = torch.cat([x['correct'] for x in outputs], dim=0)
        auc = roc_auc_score(y_true=correct.detach().cpu().numpy(), y_score=pred_probas.detach().cpu().numpy())
        tensorboard_logs = {'{}/loss'.format(mode): avg_loss, '{}/likelihood'.format(mode): avg_likelihood,
                            '{}/accuracy'.format(mode): avg_accuracy, '{}/roc_auc'.format(mode): auc,
                            '{}/predicted_label_proba'.format(mode): avg_pred_proba}
        return {'{}_loss'.format(short_name): avg_loss, '{}_likelihood'.format(short_name): avg_likelihood,
                '{}_accuracy'.format(short_name): avg_accuracy, '{}_auc'.format(short_name): auc,
                '{}_pred_proba'.format(short_name): avg_pred_proba, 'log': tensorboard_logs}

    def configure_optimizers(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda i: 1 / np.sqrt(i + 1))
        # can return multiple optimizers and learning_rate schedulers
        return [self.optimizer], [scheduler]

    def training_epoch_end(self, outputs):
        return {'loss': np.mean([x['loss'] for x in outputs]),
                'likelihood': np.mean([x['likelihood'] for x in outputs]),
                'accuracy': np.mean([x['accuracy'] for x in outputs])}
