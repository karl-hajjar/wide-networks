import torch

ACTIVATION_DICT = {'relu': torch.nn.ReLU,
                   'elu': torch.nn.ELU,
                   'gelu': torch.nn.GELU,
                   # 'tanh': torch.nn.modules.activation.Tanh,
                   'identity': torch.nn.Identity}
DEFAULT_ACTIVATION = "relu"


def get_activation(activation=None):
    if activation is None:
        return ACTIVATION_DICT[DEFAULT_ACTIVATION]
    elif isinstance(activation, str):
        if activation in ACTIVATION_DICT.keys():
            return ACTIVATION_DICT[activation]
        else:
            raise ValueError("Activation name must be one of {} but was {}".format(list(ACTIVATION_DICT.keys()),
                                                                                   activation))
    elif isinstance(activation, torch.nn.Module):
        return activation
    else:
        raise ValueError("activation argument must be of type None, str, or torch.nn.Module but was of type {}".\
                         format(type(activation)))

