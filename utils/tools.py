import os
import logging
import numpy as np
import torch
import yaml
import pickle


def set_up_logging(path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] -- %(module)s - %(funcName)s  %(message)s",
        handlers=[
            logging.FileHandler(path, mode='w'),
            logging.StreamHandler()]
    )


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_yaml(path):
    with open(path, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise Exception("Exception while reading yaml file {} : {}".format(d, e))
    return d


def create_dir(path):
    """
    Creates a directory if it does not exist.
    :param string path: the path at which to create the directory. According to the specifications of os.makedirs, all
    the intermediate directories will be created if needed.
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_pickle(path, single=False):
    results = []
    with open(path, "rb") as f:
        if single:
            results = pickle.load(f)
        else:
            while True:
                try:
                    results.append(pickle.load(f))
                except EOFError:
                    break
    return results
