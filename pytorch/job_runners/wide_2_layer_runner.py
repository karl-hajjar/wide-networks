import os
import logging
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle

from .job_runner import JobRunner
from utils.tools import *
from pytorch.configs.model import ModelConfig
from pytorch.models.wide_2_layer import TwoLayerNet
from utils.data import random, dataset

TRAIN_RATIO = 0.8
MAX_EPOCHS = 10000
MAX_STEPS = int(3.5e5)

Ks = [3, 4, 5, 6]
Rs = [0.5, 1.0, 2.0]
N_TRAINs = [5000, 10000, 1000, 512, 256, 128, 64]
Ds = [512, 1000, 5000, 10000, 256, 128, 50, 25, 20, 15, 10, 5]
Ms = [10, 50, 100, 200, 500, 1000, 2000, 5000]
BATCH_SIZES = [128, 64, 32, 1]


class Wide2LayerRunner(JobRunner):
    """
    A class to run multiple trials of different experiments on the wide 2-layer networks.
    """

    def __init__(self, config_dict: dict, base_experiment_path: str, n_rep):
        super().__init__(config_dict, base_experiment_path)
        self.original_config_dict = config_dict  # keep a copy of the original config_dict because of later changes
        self.n_rep = n_rep
        if ('initializer' in config_dict.keys()) and ('params' in config_dict['initializer'].keys()) and \
           ('std' in config_dict['initializer']['params'].keys()):
            self.model_config += '_std={:.3f}'.format(config_dict['initializer']['params']['std'])

        self.exp_dir = ''
        self.version = ''

        set_random_seeds(self.SEED)  # set random seeds of numpy and pytorch for reproducibility
        self.trial_seeds = np.random.randint(0, 100, size=n_rep)  # define random seeds to use for each trial

    def run(self):
        for k in Ks:
            for r in Rs:
                for batch_size in BATCH_SIZES:
                    for n_train in N_TRAINs:
                        for d in Ds:
                            for m in Ms:
                                n = int(n_train / TRAIN_RATIO)
                                n_val = (n - n_train) // 2
                                self._prepare_experiment(k, r, batch_size, n_train, d, m)
                                self._prepare_data_for_trials(k, r, batch_size, n, n_train, n_val, d)
                                for idx in range(self.n_rep):
                                    self._run_trial(idx, self.trial_seeds[idx], k, r, batch_size, n, n_train, d, m)

    def _prepare_experiment(self, k, r, batch_size, n_train, d, m):
        self.config_dict['architecture']['input_size'] = d
        self.config_dict['architecture']['hidden_layer_dim'] = m
        exp_name = 'bsize={}_ntrain={}_d={}_m={}'.format(batch_size, n_train, d, m)
        k_r_config = 'k={}_r={}'.format(k, r)
        self.exp_dir = os.path.join(self.base_experiment_path, k_r_config, exp_name)  # final experiment folder
        self.version = '{}_{}_{}'.format(self.model_config, k_r_config, exp_name)

    def _prepare_data_for_trials(self, k, r, batch_size, n, n_train, n_val, d):
        set_random_seeds(self.SEED)  # set random seed back to its original value as it is modified in each trial
        ds = self._generate_data(k, r, d, n)  # generate data in the form of a PyTorch Dataset
        self._set_data_loaders(ds, n, n_train, n_val, batch_size)  # define data loaders

    def _run_trial(self, idx, seed, k, r, batch_size, n, n_train, d, m):
        trial_name = 'trial_{}'.format(idx + 1)
        self.trial_dir = os.path.join(self.exp_dir, trial_name)

        if not os.path.exists(self.trial_dir):  # run trial only if it doesn't already exist
            create_dir(self.trial_dir)  # directory to save the trial
            self.trial_version = '{}_{}'.format(self.version, trial_name)  # version for TensorBoard

            self._set_tb_logger_and_callbacks(trial_name)  # tb logger, checkpoints and early stopping

            log_dir = os.path.join(self.trial_dir, self.LOG_NAME)  # define path to save the logs of the trial
            set_up_logging(log_dir)
            logging.info('----- Trial {:,} with version {} -----\n'.format(idx, self.trial_version))
            self._log_experiment_info(k, r, batch_size, n, n_train, d, m)

            set_random_seeds(seed)  # set random seed for the trial
            logging.info('Random seed used for the script : {:,}'.format(self.SEED))
            logging.info('Random seed used for the trial : {:,}\n'.format(seed))

            config = ModelConfig(config_dict=self.config_dict)  # define the config as a class to pass to the model
            two_layer_net = TwoLayerNet(config, train_hidden=True)  # define the model
            logging.info('Number of model parameters : {:,}'.format(two_layer_net.count_parameters()))
            logging.info('Model architecture :\n{}\n'.format(two_layer_net))

            # training and validation pipeline
            trainer = pl.Trainer(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS, logger=self.tb_logger,
                                 checkpoint_callback=self.checkpoint_callback,
                                 num_sanity_val_steps=0, early_stop_callback=self.early_stopping_callback)
            trainer.fit(model=two_layer_net, train_dataloader=self.train_data_loader,
                        val_dataloaders=self.val_data_loader)

            # test pipeline
            test_results = trainer.test(model=two_layer_net, test_dataloaders=self.test_data_loader)
            logging.info('Test results :\n{}\n'.format(test_results))

            # save all training val and test results to pickle file
            with open(os.path.join(self.trial_dir, self.RESULTS_FILE), 'wb') as file:
                pickle.dump(two_layer_net.results, file)

    def _log_experiment_info(self, k, r, batch_size, n, n_train, d, m):
        logging.info('Square root of the number of clusters k = {:,}'.format(k))
        logging.info('Max euclidean norm of the data r = {:,}'.format(r))
        logging.info('Batch size = {:,}'.format(batch_size))
        logging.info('Total number of data points n = {:,}'.format(n))
        logging.info('Number training samples n_train = {:,}'.format(n_train))
        logging.info('d = {:,}'.format(d))
        logging.info('m = {:,}'.format(m))
        logging.info('activation : {}'.format(self.config_dict['activation']['name']))
        logging.info('loss : {}'.format(self.config_dict['loss']['name']))
        logging.info('optimizer : {}'.format(self.config_dict['optimizer']['name']))

        if ('initializer' in self.config_dict.keys()) and ('name' in self.config_dict['initializer'].keys()):
            initializer = self.config_dict['initializer']['name']
        else:
            initializer = 'custom'
        logging.info('initializer : {}'.format(initializer))

        if ('normalization' in self.config_dict.keys()) and ('name' in self.config_dict['normalization'].keys()):
            norm = self.config_dict['initializer']['name']
        else:
            norm = 'None'
        logging.info('normalization : {}\n'.format(norm))

    @staticmethod
    def _generate_data(k, r, d, n) -> torch.utils.data.Dataset:
        data = random.RandomData(k=k, r=r, d=d, n=n)
        data.generate_samples()
        return dataset.RandomDataset(data)

    def _set_data_loaders(self, ds, n, n_train, n_val, batch_size):
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

        self.train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
        self.val_data_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
        self.test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    def _set_tb_logger_and_callbacks(self, trial_name):
        """
        Define TensorBoard logger and checkpoint callbacks.
        :return:
        """
        self.tb_logger = TensorBoardLogger(save_dir=self.exp_dir, version=trial_name, name=None)
        checkpoints_name_template = '{epoch}_{val_accuracy:.3f}_{val_loss:.3f}_{val_auc:.3f}'
        checkpoints_path = os.path.join(self.trial_dir, 'checkpoints', checkpoints_name_template)
        self.checkpoint_callback = ModelCheckpoint(filepath=checkpoints_path,
                                                   save_top_k=3,
                                                   verbose=True,
                                                   monitor='val_accuracy',
                                                   mode='max',
                                                   prefix='')
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=5, mode='min')
