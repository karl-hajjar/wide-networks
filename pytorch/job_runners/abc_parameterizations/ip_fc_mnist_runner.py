from torch.utils.data import Dataset, Subset

from .abc_runner import ABCRunner


# class IpFcMNIST(ABCRunner):
#     """
#     A class to run the IP parameterization on the MNIST dataset.
#     """
#     def __init__(self, config_dict: dict, base_experiment_path: str, train_dataset: Dataset, test_dataset: Dataset,
#                  val_dataset: Dataset = None, train_ratio: float = 0.8, n_trials: int = 10):
#         super().__init__(config_dict, base_experiment_path, train_dataset, test_dataset, val_dataset, train_ratio,
#                          n_trials)

