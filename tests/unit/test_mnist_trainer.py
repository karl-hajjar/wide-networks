import unittest
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from pytorch.models.resnet_mnist import ResNetMNIST
from pytorch.configs.model import ModelConfig

RESOURCES_DIR = '../resources/'
BATCH_SIZE = 32
DATA_DIR = '../../data/'
N_SAMPLES = 6000
RATIO_TRAIN = 0.8
SAVE_DIR = '../../experiments'
NAME = 'resnet'
N_BATCHES = 200
PRECISION = 1e-6

version = 'test_lr_scheduler_1.001'

# Run
# ```tensorboard --logdir path_to_the_exp_version```
# Alternatively, one can run
# tensorboard --logdir_spec=name1:./bsize=1_ntrain=64_d=256_m=100,name2:./bsize=1_ntrain=64_d=256_m=1000
# to specicfy multiple log directories
# in the command line to and then open http://localhost:6006/ to see the TensorBoard


class TestMNISTTrainer(unittest.TestCase):
    def setUp(self) -> None:
        # define model
        config = ModelConfig(config_file=os.path.join(RESOURCES_DIR, 'resnet_config.yaml'))
        self.resnet = ResNetMNIST(config)

        # set up train and val data loaders
        n_train = int(RATIO_TRAIN * N_SAMPLES)
        n_val = N_SAMPLES - n_train
        indexes = list(range(N_SAMPLES))
        np.random.shuffle(indexes)
        train_indexes = indexes[:n_train]
        val_indexes = indexes[n_train:]

        self.train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                              indexes=train_indexes)

        self.val_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                            indexes=val_indexes)

        self.test_dataloader = self.resnet.test_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE)

        self.tb_logger = TensorBoardLogger(save_dir=SAVE_DIR, version=version, name=NAME)
        checkpoints_path = os.path.join(SAVE_DIR, NAME, version, 'checkpoints',
                                        '{epoch}_{val_accuracy:.3f}_{val_loss:.3f}_{val_auc:.3f}')
        self.checkpoint_callback = ModelCheckpoint(
                                        filepath=checkpoints_path,
                                        save_top_k=3,
                                        verbose=True,
                                        monitor='val_accuracy',
                                        mode='max',
                                        prefix=''
                                    )

    def test_trainer_fit(self):
        trainer = pl.Trainer(max_epochs=15, logger=self.tb_logger, checkpoint_callback=self.checkpoint_callback,
                             num_sanity_val_steps=0)
        trainer.fit(model=self.resnet, train_dataloader=self.train_data_loader, val_dataloaders=self.val_data_loader)

    def test_likelihood_shapes(self):
        # configure new data loaders with batch_size = 1 for comparing exp(-loss) to likelihood
        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                         indexes=list(range(500)))
        val_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                       indexes=list(range(500, 1000)))

        # on training batches
        for id, batch in enumerate(train_data_loader):
            if id >= N_BATCHES:
                break
            x, y = batch
            y_hat = self.resnet(x)
            all_probas = self.resnet.predict(y_hat, mode='probas', from_logits=True)
            pred_proba, pred_label = torch.max(all_probas, 1)
            # get probability of target labels only and then average over batch
            ll = all_probas[:, y]
            self.assertTrue(ll.shape == (len(all_probas), len(y)))
            likelihood = ll.mean()
            true_likelihood = all_probas[torch.range(0, len(y)-1, dtype=torch.long), y]
            self.assertTrue(true_likelihood.shape == (len(all_probas),))
            true_likelihood = true_likelihood.mean()
            self.assertTrue(abs(true_likelihood - likelihood) > 1.0e-4)

        # on val batches
        for id, batch in enumerate(val_data_loader):
            if id >= N_BATCHES:
                break
            x, y = batch
            y_hat = self.resnet(x)
            all_probas = self.resnet.predict(y_hat, mode='probas', from_logits=True)
            pred_proba, pred_label = torch.max(all_probas, 1)
            # get probability of target labels only and then average over batch
            ll = all_probas[:, y]
            self.assertTrue(ll.shape == (len(all_probas), len(y)))
            likelihood = ll.mean()
            true_likelihood = all_probas[torch.range(0, len(y)-1, dtype=torch.long), y]
            self.assertTrue(true_likelihood.shape == (len(all_probas),))
            true_likelihood = true_likelihood.mean()
            self.assertTrue(abs(true_likelihood - likelihood) > 1.0e-4)

    def test_likelihood_values(self):
        # configure new data loaders with batch_size = 1 for comparing exp(-loss) to likelihood
        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=1,
                                                         indexes=list(range(500)))
        val_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=1,
                                                       indexes=list(range(500, 1000)))
        # on training batches
        for id, batch in enumerate(train_data_loader):
            if id >= N_BATCHES:
                break
            output = self.resnet.training_step(batch, id)
            exp_loss = torch.exp(-output["loss"])  # loss is the negative log-likelihood
            ll = output["likelihood"]
            exp_loss_vs_ll = (exp_loss - ll).abs().mean()
            self.assertTrue(exp_loss_vs_ll < PRECISION)

        # on val bacthes
        for id, batch in enumerate(val_data_loader):
            if id >= N_BATCHES:
                break
            output = self.resnet.training_step(batch, id)
            exp_loss = torch.exp(-output["loss"])  # loss is the negative log-likelihood
            ll = output["likelihood"]
            exp_loss_vs_ll = (exp_loss - ll).abs().mean()
            self.assertTrue(exp_loss_vs_ll < PRECISION)

    def test_probability_values(self):
        train_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                         indexes=list(range(500)))
        val_data_loader = self.resnet.train_dataloader(data_dir=DATA_DIR, download=False, batch_size=BATCH_SIZE,
                                                       indexes=list(range(500, 1000)))
        # on training batches
        for id, batch in enumerate(train_data_loader):
            if id >= N_BATCHES:
                break
            x, y = batch
            y_hat = self.resnet(x)
            # probas from logits
            all_probas_from_logits = self.resnet.predict(y_hat, mode='probas', from_logits=True)
            torch.testing.assert_allclose(all_probas_from_logits.sum(dim=1), 1.0, rtol=PRECISION, atol=PRECISION)
            # probas from input
            all_probas_from_input = self.resnet.predict(x, mode='probas', from_logits=False)
            torch.testing.assert_allclose(all_probas_from_input, all_probas_from_logits, rtol=PRECISION, atol=PRECISION)
            torch.testing.assert_allclose(all_probas_from_input.sum(dim=1), 1.0, rtol=PRECISION, atol=PRECISION)

        # on val bacthes
        for id, batch in enumerate(val_data_loader):
            if id >= N_BATCHES:
                break
            x, y = batch
            y_hat = self.resnet(x)
            # probas from logits
            all_probas_from_logits = self.resnet.predict(y_hat, mode='probas', from_logits=True)
            torch.testing.assert_allclose(all_probas_from_logits.sum(dim=1), 1.0, rtol=PRECISION, atol=PRECISION)
            # probas from input
            all_probas_from_input = self.resnet.predict(x, mode='probas', from_logits=False)
            torch.testing.assert_allclose(all_probas_from_input, all_probas_from_logits, rtol=PRECISION, atol=PRECISION)
            torch.testing.assert_allclose(all_probas_from_input.sum(dim=1), 1.0, rtol=PRECISION, atol=PRECISION)

    def test_trainer_test_method(self):
        trainer = pl.Trainer(max_epochs=5, logger=self.tb_logger, num_sanity_val_steps=0,
                             checkpoint_callback=self.checkpoint_callback)

        fit_results = trainer.fit(model=self.resnet, train_dataloader=self.train_data_loader, val_dataloaders=self.val_data_loader)
        print('type(test_results) :', type(fit_results))
        print('test_results :\n', fit_results)

        test_results = trainer.test(model=self.resnet, test_dataloaders=self.test_dataloader)
        print('type(test_results) :', type(test_results))
        print('test_results :\n', test_results)


if __name__ == '__main__':
    unittest.main()
