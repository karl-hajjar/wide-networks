import torch


class ExponentialLoss(torch.nn.Module):
    """
    A class defining the exponential loss for binary classification:
    For a prediction y_hat and a target y in {-1, 1}, the exponential loss is equal to:
        l = exp(-y_hat * y)
    """
    def __init__(self, reduction='mean'):
        super(ExponentialLoss, self).__init__()
        if reduction not in ['sum', 'mean', None]:
            raise ValueError("'reduction' argument must be one of {} but was {}".format(['sum', 'mean', None],
                                                                                        reduction))
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        l = torch.exp(- prediction * target)  # the values of target are expected to be in {-1, 1}
        if self.reduction == 'sum':
            l = l.sum()
        elif self.reduction == 'mean':
            l = l.mean()
        return l


LOSS_DICT = {'cross_entropy': torch.nn.CrossEntropyLoss,
             'kl': torch.nn.KLDivLoss,
             'mse': torch.nn.MSELoss,
             'logistic': torch.nn.BCEWithLogitsLoss,
             'exponential': ExponentialLoss}

DEFAULT_LOSS = "cross_entropy"


def get_loss(loss=None):
    if loss is None:
        return LOSS_DICT[DEFAULT_LOSS]
    elif isinstance(loss, str):
        if loss in LOSS_DICT.keys():
            return LOSS_DICT[loss]
        else:
            raise ValueError("Loss name must be one of {} but was {}".format(list(LOSS_DICT.keys()),
                                                                             loss))
    elif isinstance(loss, torch.nn.Module):
        return loss
    else:
        raise ValueError("loss argument must be of type None, str, or torch.nn.Module but was of type {}".\
                         format(type(loss)))
