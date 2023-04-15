import os
import logging
import numpy as np
import torch
import yaml
import pickle


def set_up_logger(path: str):
    # first remove handlers if there were some already defined
    logger = logging.getLogger()  # root logger
    for handler in logger.handlers:  # remove all old handlers
        handler.close()
        logger.removeHandler(handler)

    # set new handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] -- %(module)s - %(funcName)s  %(message)s")
    file_handler = logging.FileHandler(path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add new handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


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


def get_best_model_checkpoint(path):
    best_val_accuracy = 0.0
    best_checkpoint = None
    for f in os.scandir(path):
        val_accuracy = float(f.name.split('val_accuracy=')[1].split('_')[0])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = f.name
    return best_checkpoint, best_val_accuracy
