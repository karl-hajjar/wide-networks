from torch.utils.data import Dataset

from pytorch.job_runners.job_runner import JobRunner


class IpFcRunner(JobRunner):
    """
    A class to run the IP parameterization on the MNIST dataset.
    """

    TRAIN_RATIO = 0.8

    def __init__(self, config_dict: dict, base_experiment_path: str, train_dataset: Dataset, test_dataset: Dataset,
                 val_dataset: Dataset = None, n_trials: int = 10):
        super().__init__(config_dict, base_experiment_path)
        if val_dataset is None:
            self._set_train_val_data_from_train(train_dataset)
        self.test_dataset = test_dataset
        pass

    def _set_model_config(self, config_dict):
        pass

    def _set_train_val_data_from_train(self, train_dataset):
        # self.train_dataset =
        # self.val_dataset =
        pass

    def run(self):
        pass
