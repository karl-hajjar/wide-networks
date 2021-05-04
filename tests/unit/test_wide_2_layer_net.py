import unittest
import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from pytorch.configs.model import ModelConfig
from pytorch.models.wide_2_layer import TwoLayerNet
from utils.dataset import random, dataset
from utils.nn import smoothed_exp_margin, smoothed_logistic_margin

BATCH_SIZE = 32
PRECISION = 1.0e-5


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_file = os.path.join('../../pytorch/configs', 'wide_two_layer_net.yaml')
        with open(config_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise Exception("Exception while reading yaml file {} : {}".format(config_file, e))
        self.config = ModelConfig(config_dict=config_dict)
        self.two_layer_net = TwoLayerNet(self.config, train_hidden=True)

        r, k, n = 1., 4, 5000
        d = config_dict['architecture']['input_size']

        data = random.RandomData(k=k, r=r, d=d, n=n)
        data.generate_samples()
        self.ds = dataset.RandomDataset(data)
        self.data_loader = DataLoader(self.ds, shuffle=True, batch_size=BATCH_SIZE)

    def test_custom_init(self):
        config = deepcopy(self.config)
        config.initializer = None
        two_layer_net = TwoLayerNet(config, train_hidden=True)
        for i, p in enumerate(two_layer_net.layer2.parameters()):  # each output unit should be in {-1, 1}
            self.assertTrue((p == 1.).sum() + (p == -1.).sum() == two_layer_net.hidden_layer_dim)
        self.assertTrue((two_layer_net.beta - 2.0).abs() < PRECISION)

    def test_output_shape(self):
        for (x, y) in self.data_loader:
            out = self.two_layer_net.forward(x)
            self.assertTrue(len(out.shape) == 2)
            self.assertSequenceEqual(out.shape, (len(x), 1))
            self.assertSequenceEqual(out.shape, y.shape)

    def test_metrics_computation(self):
        all_sample_margins = []
        all_margins = []
        for (x, y) in self.data_loader:
            y_0_1 = (y + 1) / 2  # converting target labels from {-1, 1} to {0, 1}.
            y_hat = self.two_layer_net(x)
            loss = self.two_layer_net.loss(y_hat, y_0_1)

            sample_margins = y * y_hat  # y assumed to have values in {-1, 1}
            all_sample_margins.append(sample_margins)
            self.assertSequenceEqual(sample_margins.shape, (len(x), 1))

            likelihood = self.two_layer_net.predict(sample_margins, mode='probas', from_logits=True)
            self.assertSequenceEqual(likelihood.shape, (len(x), 1))
            self.assertTrue((likelihood <= 1).all())
            self.assertTrue((likelihood >= 0).all())

            pos_probas = self.two_layer_net.predict(y_hat, mode='probas', from_logits=True)
            self.assertSequenceEqual(pos_probas.shape, (len(x), 1))
            self.assertTrue((pos_probas <= 1).all())
            self.assertTrue((pos_probas >= 0).all())

            self.assertTrue(((likelihood[y == 1] - pos_probas[y == 1]).mean().abs() < PRECISION).all())
            # neg_probas = 1 - pos_probas
            self.assertTrue(((likelihood[y == -1] - 1 + pos_probas[y == -1]).mean().abs() < PRECISION).all())

            pred_label = (pos_probas >= 0.5).long()
            self.assertTrue((pred_label == 1).sum() + (pred_label == 0).sum() == len(x))

            pred_proba = torch.max(pos_probas, 1 - pos_probas)  # element-wise max of the 2 tensors
            self.assertTrue((pred_proba >= 0.5).all())
            self.assertTrue((pred_proba <= 1.0).all())

            acc = (pred_label == y_0_1).sum() / float(len(y))
            self.assertTrue(acc >= 0.)
            self.assertTrue(acc <= 1.)

            # margin and weights norm
            margin = sample_margins.min()
            all_margins.append(margin)
            self.assertSequenceEqual(margin.shape, ())

            beta = self.two_layer_net.beta
            self.assertSequenceEqual(beta.shape, ())
            self.assertTrue(beta.detach().item() > 0)

            # normalized margin should be bounded by sqrt(r^2 + 1) /2 where r is a bound on the norm of the data points
            normalized_margin = margin / beta  # see https://github.com/lchizat/2020-implicit-bias-wide-2NN/blob/3acfbb441cf8b235a9982497553a4c25d9ee6623/implicit_bias_2NN_utils.jl#L38
            self.assertSequenceEqual(normalized_margin.shape, ())
            self.assertTrue(normalized_margin.detach().item() <=
                            np.sqrt(self.data_loader.dataset.data.r ** 2 + 1) / 2)

            smoothed_exp_m = smoothed_exp_margin(beta, sample_margins)
            self.assertTrue(margin.detach().item() <= smoothed_exp_m.detach().item())
            self.assertTrue(smoothed_exp_m.detach().item() <= (np.log(len(x)) / beta + margin).detach().item())

            smoothed_log_m = smoothed_logistic_margin(beta, sample_margins)
            self.assertTrue(smoothed_exp_m.detach().item() <= smoothed_log_m.detach().item())

        self.assertTrue((torch.cat(all_sample_margins).min() - torch.stack(all_margins).min()).abs() < PRECISION)


if __name__ == '__main__':
    unittest.main()
