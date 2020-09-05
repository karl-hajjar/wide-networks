import torch

OPT_DICT = {'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop,
            'sgd': torch.optim.SGD}
DEFAULT_OPT = "adam"


def get_optimizer(optimizer=None):
    if optimizer is None:
        return OPT_DICT[DEFAULT_OPT]
    elif isinstance(optimizer, str):
        if optimizer in OPT_DICT.keys():
            return OPT_DICT[optimizer]
        else:
            raise ValueError("Optimizer name must be one of {} but was {}".format(list(OPT_DICT.keys()),
                                                                                  optimizer))
    elif isinstance(optimizer, torch.nn.Module):
        return optimizer
    else:
        raise ValueError("optimizer argument must be of type None, str, or torch.nn.Module but was of type {}".\
                         format(type(optimizer)))

