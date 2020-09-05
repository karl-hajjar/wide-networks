import torch

NORM_DICT = {'batch_norm_1d': torch.nn.BatchNorm1d,
             'batch_norm_2d': torch.nn.BatchNorm2d,
             'batch_norm_3d': torch.nn.BatchNorm3d,
             'layer_norm': torch.nn.LayerNorm}
DEFAULT_NORM = "batch_norm_2d"


def get_norm(norm=None):
    if norm is None:
        return NORM_DICT[DEFAULT_NORM]
    elif isinstance(norm, str):
        if norm in NORM_DICT.keys():
            return NORM_DICT[norm]
        else:
            raise ValueError("Normalization name must be one of {} but was {}".format(list(NORM_DICT.keys()),
                                                                                      norm))
    elif isinstance(norm, torch.nn.Module):
        return norm
    else:
        raise ValueError("optimizer argument must be of type None, str, or torch.nn.Module but was of type {}". \
                         format(type(norm)))

