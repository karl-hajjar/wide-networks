import torch

INIT_DICT = {'glorot_uniform': torch.nn.init.xavier_uniform_,
             'glorot_normal': torch.nn.init.xavier_normal_,
             'normal': torch.nn.init.normal_,
             'uniform': torch.nn.init.uniform_}
DEFAULT_INIT = "glorot_uniform"


def get_initializer(initializer=None):
    if initializer is None:
        return INIT_DICT[DEFAULT_INIT]
    elif isinstance(initializer, str):
        if initializer in INIT_DICT.keys():
            return INIT_DICT[initializer]
        else:
            raise ValueError("Initializer name must be one of {} but was {}".format(list(INIT_DICT.keys()),
                                                                                    initializer))
    elif isinstance(initializer, torch.nn.Module):
        return initializer
    else:
        raise ValueError("optimizer argument must be of type None, str, or torch.nn.Module but was of type {}". \
                         format(type(initializer)))

