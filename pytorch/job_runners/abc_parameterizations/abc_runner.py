import os
import pickle
import logging
import numpy as np

from torch.utils.data import Dataset, Subset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch.job_runners.job_runner import JobRunner
from pytorch.models.abc_params.base_abc_param import BaseABCParam
from utils.tools import create_dir, set_up_logger, set_random_seeds
from pytorch.configs.model import ModelConfig


class ABCRunner(JobRunner):
    """
    A class to run an experiment with an abc-parameterization on a given dataset.
    """

    MAX_EPOCHS = 15
    MAX_STEPS = int(1.5e3)
    BASE_LR = 0.001

    def __init__(self, config_dict: dict, base_experiment_path: str, model: BaseABCParam, train_dataset: Dataset,
                 test_dataset: Dataset, val_dataset: Dataset = None, train_ratio: float = 0.8, n_trials: int = 10,
                 early_stopping=False, calibrate_base_lr=False):
        self.width = config_dict['architecture']['width']
        self.batch_size = config_dict['training']['batch_size']
        self._set_base_lr(config_dict)

        super().__init__(config_dict, base_experiment_path)

        if 'n_epochs' in config_dict['training'].keys():
            self.max_epochs = config_dict['training']['n_epochs']
        else:
            self.max_epochs = self.MAX_EPOCHS

        if 'n_steps' in config_dict['training'].keys():
            self.max_steps = config_dict['training']['n_steps']
        else:
            self.max_steps = self.MAX_STEPS

        if val_dataset is None:
            self._set_train_val_data_from_train(train_dataset, train_ratio)
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self._set_data_loaders()
        if calibrate_base_lr:
            self.lr_calibration_batches = list(self.train_data_loader)[:2]

        self.model = model
        self.n_trials = n_trials

        if 'early_stopping' in config_dict['training'].keys():
            self.early_stopping = config_dict['training']['early_stopping']
        else:
            self.early_stopping = early_stopping

        self.early_stopping_callback = False  # this is modified in _set_tb_logger_and_callbacks in early_stopping=True

        set_random_seeds(self.SEED)  # set random seed for reproducibility
        self.trial_seeds = np.random.randint(0, 100, size=n_trials)  # define random seeds to use for each trial

    def _set_model_version(self, config_dict):
        # recall L is the number of HIDDEN layers
        self.model_version = 'L={}_m={}'.format(config_dict['architecture']['n_layers'] - 1,
                                                config_dict['architecture']['width'])

    def _set_model_config(self, config_dict):
        # define string to represent model
        model_config = 'activation={}_lr={}_batchsize={}_bias={}'.format(config_dict['activation']['name'],
                                                                         self.base_lr,
                                                                         self.batch_size,
                                                                         config_dict['architecture']['bias'])
        if ('params' in config_dict['optimizer'].keys()) and \
           ('weight_decay' in config_dict['optimizer']['params'].keys()):
            model_config += '_' + 'wd={}'.format(config_dict['optimizer']['params']['weight_decay'])
        if ('scheduler' in config_dict.keys()) and ('params' in config_dict['scheduler'].keys()) and \
                ('n_warmup_steps' in config_dict['scheduler']['params'].keys()):
            model_config += '_' + 'warmup={}'.format(config_dict['scheduler']['params']['n_warmup_steps'])
        self.model_config = model_config

    def _set_base_lr(self, config_dict):
        if ('params' in config_dict['optimizer'].keys()) and ('lr' in config_dict['optimizer']['params'].keys()):
            self.base_lr = config_dict['optimizer']['params']['lr']
        else:
            self.base_lr = self.BASE_LR

    def _set_train_val_data_from_train(self, train_dataset, train_ratio):
        n = len(train_dataset)
        n_train = int(train_ratio * n)
        train_indexes = range(n_train)
        val_indexes = range(n_train, n)
        self.train_dataset = Subset(train_dataset, train_indexes)
        self.val_dataset = Subset(train_dataset, val_indexes)

    def _set_data_loaders(self):
        self.train_data_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.val_data_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size)
        self.test_data_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

    def run(self, **kwargs):
        for idx in range(self.n_trials):
            self._run_trial(idx, **kwargs)

    def _run_trial(self, idx, **kwargs):
        if hasattr(self, "lr_calibration_batches"):
            kwargs['lr_calibration_batches'] = self.lr_calibration_batches
        trial_name = 'trial_{}'.format(idx + 1)
        self.trial_dir = os.path.join(self.base_experiment_path, trial_name)  # folder to hold trial results

        # TODO: if os.path.exists(self.trial_dir), read lines of corresponding log file and check if last line has
        #  "test" in it. If not, re-run the trial because it means there was a problem with the run.

        if os.path.exists(self.trial_dir):
            logging.warning("Directory for trial {:,} of experiment {} already exists".format(idx, self.model_config))

        else:  # run trial only if it doesn't already exist
            create_dir(self.trial_dir)  # directory to save the trial
            set_random_seeds(self.trial_seeds[idx])  # set random seed for the trial

            self._set_tb_logger_and_callbacks(trial_name)  # tb logger, checkpoints and early stopping

            log_dir = os.path.join(self.trial_dir, self.LOG_NAME)  # define path to save the logs of the trial
            logger = set_up_logger(log_dir)

            config = ModelConfig(config_dict=self.config_dict)  # define the config as a class to pass to the model
            model = self.model(config, **kwargs)  # define the model

            logger.info('----- Trial {:,} ----- with model config {}\n'.format(idx + 1, self.model_config))
            self._log_experiment_info(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset), model.std)
            logger.info('Random seed used for the script : {:,}'.format(self.SEED))
            logger.info('Number of model parameters : {:,}'.format(model.count_parameters()))
            logger.info('Model architecture :\n{}\n'.format(model))

            try:
                trainer = pl.Trainer(max_epochs=self.max_epochs, max_steps=self.max_steps, logger=self.tb_logger,
                                     checkpoint_callback=self.checkpoint_callback, num_sanity_val_steps=0,
                                     early_stop_callback=self.early_stopping_callback)
                trainer.fit(model=model, train_dataloader=self.train_data_loader, val_dataloaders=self.val_data_loader)

                # test pipeline
                test_results = trainer.test(model=model, test_dataloaders=self.test_data_loader)
                logger.info('Test results :\n{}\n'.format(test_results))

                # save all training, val and test results to pickle file
                results_path = os.path.join(self.trial_dir, self.RESULTS_FILE)
                logging.info("Dumping results at {}".format(results_path))
                with open(results_path, 'wb') as file:
                    pickle.dump(model.results, file)

            except Exception as e:
                # dump and save results before exiting
                with open(os.path.join(self.trial_dir, self.RESULTS_FILE), 'wb') as file:
                    pickle.dump(model.results, file)
                logger.warning('model results dumped before interruption')
                logger.exception("Exception while running the train-val-test pipeline : {}".format(e))
                raise Exception(e)

    def _set_tb_logger_and_callbacks(self, trial_name):
        """
        Define TensorBoard logger and checkpoint callbacks.
        :return:
        """
        self.tb_logger = TensorBoardLogger(save_dir=self.base_experiment_path, version=trial_name, name=None)
        checkpoints_name_template = '{epoch}_{val_accuracy:.3f}_{val_loss:.3f}'
        checkpoints_path = os.path.join(self.trial_dir, 'checkpoints', checkpoints_name_template)
        self.checkpoint_callback = ModelCheckpoint(filepath=checkpoints_path,
                                                   save_top_k=3,
                                                   verbose=True,
                                                   monitor='val_accuracy',
                                                   mode='max',
                                                   prefix='')

        if self.early_stopping:
            self.early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=1.0e-6, patience=5, mode='min')

    def _log_experiment_info(self, n_train, n_val, n_test, std):
        logger = logging.getLogger()
        logger.info('Batch size = {:,}'.format(self.batch_size))
        logger.info('Base learning rate = {:,}'.format(self.base_lr))
        logger.info('Number training samples = {:,}'.format(n_train))
        logger.info('Number validation samples = {:,}'.format(n_val))
        logger.info('Number test samples = {:,}'.format(n_test))
        logger.info('width = {:,}'.format(self.width))
        logger.info('activation : {}'.format(self.config_dict['activation']['name']))
        logger.info('bias = {}'.format(self.config_dict['architecture']['bias']))
        logger.info('loss : {}'.format(self.config_dict['loss']['name']))
        logger.info('initialization std : {}'.format(std))
