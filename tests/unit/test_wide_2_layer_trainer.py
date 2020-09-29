import unittest
import os
import yaml
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pickle

from pytorch.configs.model import ModelConfig
from pytorch.models.wide_2_layer import TwoLayerNet
from utils.data import random, dataset

RESOURCES_DIR = '../resources/'
BATCH_SIZE = 64
DATA_DIR = '../../data/'
SAVE_DIR = '../../experiments'
NAME = 'wide_2_layer'
PRECISION = 1e-5
RESULTS_FILE = 'results.pickle'

MAX_EPOCHS = 5000
MAX_STEPS = int(1.5e5)

version = 'test_trainer_n=256'


class TestWide2LayerTrainer(unittest.TestCase):
    def setUp(self) -> None:
        config_file = os.path.join('../../pytorch/configs', 'wide_two_layer_net.yaml')
        with open(config_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise Exception("Exception while reading yaml file {} : {}".format(config_file, e))

        # parameters of the experiment
        r, k, self.n, d = 0.5, 3, 700, 20
        self.n_train, self.n_val = 256, 256  # n_test = n - (n_train + n_val)

        self.version = 'test_new_mac_n={}_d={}_m={}'.\
            format(self.n, d, config_dict['architecture']['hidden_layer_dim'])

        # config and net
        config_dict['architecture']['input_size'] = d
        self.config = ModelConfig(config_dict=config_dict)
        self.two_layer_net = TwoLayerNet(self.config, train_hidden=True)

        # generate data
        ds = self._generate_data(k, r, d, self.n)

        # define train/val/test data loaders
        self._set_data_loaders(ds, self.n, self.n_train, self.n_val)

        # define tb logger and callbacks
        self.tb_logger = TensorBoardLogger(save_dir=SAVE_DIR, version=self.version, name=NAME)
        checkpoints_name_template = '{epoch}_{val_accuracy:.3f}_{val_loss:.3f}_{val_auc:.3f}'
        checkpoints_path = os.path.join(SAVE_DIR, NAME, self.version, 'checkpoints', checkpoints_name_template)
        self.checkpoint_callback = ModelCheckpoint(filepath=checkpoints_path,
                                                   save_top_k=3,
                                                   verbose=True,
                                                   monitor='val_accuracy',
                                                   mode='max',
                                                   prefix='')
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=5, mode='min')

    @staticmethod
    def _generate_data(k, r, d, n):
        data = random.RandomData(k=k, r=r, d=d, n=n)
        data.generate_samples()
        return dataset.RandomDataset(data)

    def _set_data_loaders(self, ds, n, n_train, n_val):
        # define train/val/test indexes
        shuffled_indexes = np.arange(n)
        np.random.shuffle(shuffled_indexes)
        train_indexes = shuffled_indexes[:n_train]
        val_indexes = shuffled_indexes[n_train: n_train + n_val]
        test_indexes = shuffled_indexes[n_train + n_val:]

        # define datasets and data loaders
        training_dataset = Subset(ds, train_indexes)
        val_dataset = Subset(ds, val_indexes)
        test_dataset = Subset(ds, test_indexes)

        self.train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE)
        self.val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)
        self.test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

    def test_train_val_pipeline(self):
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=self.tb_logger, checkpoint_callback=self.checkpoint_callback,
                             num_sanity_val_steps=0, early_stop_callback=False)
        trainer.fit(model=self.two_layer_net, train_dataloader=self.train_data_loader,
                    val_dataloaders=self.val_data_loader)

    def test_trainer_test_method(self):
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=self.tb_logger, checkpoint_callback=self.checkpoint_callback,
                             num_sanity_val_steps=0, early_stop_callback=False)
        trainer.fit(model=self.two_layer_net, train_dataloader=self.train_data_loader,
                    val_dataloaders=self.val_data_loader)

        test_results = trainer.test(model=self.two_layer_net, test_dataloaders=self.test_data_loader)
        print('train results :\n{}'.format(test_results))

    def test_saving_and_loading_results(self):
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS, logger=self.tb_logger,
                             num_sanity_val_steps=0, checkpoint_callback=self.checkpoint_callback,
                             early_stop_callback=self.early_stopping_callback)
        trainer.fit(model=self.two_layer_net, train_dataloader=self.train_data_loader,
                    val_dataloaders=self.val_data_loader)

        test_results = trainer.test(model=self.two_layer_net, test_dataloaders=self.test_data_loader)
        print('train results :\n{}'.format(test_results))

        results_path = os.path.join(SAVE_DIR, NAME, self.version, RESULTS_FILE)
        with open(results_path, 'wb') as file:
            pickle.dump(self.two_layer_net.results, file)

        with open(results_path, 'rb') as file:
            results = pickle.load(file)

        self.assertTrue(len(results['training']) == len(results['validation']) == self.two_layer_net.current_epoch + 1)
        self.assertTrue(len(results['test']) == 1)
        print(results['test'][0])

        self._check_results(results['training'], mode='train')
        self._check_results(results['validation'], mode='val')
        self._check_results(results['test'], mode='test')

    def _check_results(self, results, mode):
        for res in results:
            self.assertTrue(type(res) == dict)
            keys = ['loss', 'likelihood', 'accuracy', 'margin', 'smoothed_exp_margin', 'smoothed_logistic_margin',
                    'normalized_margin']
            if mode == 'train':
                keys.append('beta')
            for key in keys:
                self.assertTrue(key in res.keys())
            self.assertTrue(res['loss'] > 0.)
            self.assertTrue(res['likelihood'] > 0.)
            self.assertTrue(res['likelihood'] <= 1.)
            self.assertTrue(res['accuracy'] > 0.)
            self.assertTrue(res['accuracy'] <= 1.)

            # bound on the normalized margin is sqrt(r^2 + 1) / 2
            self.assertTrue(res['normalized_margin'] <=
                            np.sqrt(self.train_data_loader.dataset.dataset.data.r ** 2 + 1) / 2)
            if mode == 'train':
                beta = res['beta']
                self.assertTrue(beta > 0)
                self.beta = beta
                n = self.n_train
            elif mode == 'val':
                n = self.n_val
            else:
                n = self.n - (self.n_train + self.n_val)
            self.assertTrue(res['margin'] <= res['smoothed_exp_margin'])
            self.assertTrue(res['smoothed_exp_margin'] <= (np.log(n) / self.beta + res['margin']))
            self.assertTrue(res['smoothed_exp_margin'] <= res['smoothed_logistic_margin'] + PRECISION)


if __name__ == '__main__':
    unittest.main()
