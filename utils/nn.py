import torch
import numpy as np


def get_output_size_conv(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def get_same_padding_conv(input_size, kernel_size, stride):
    return ((stride - 1) * input_size + kernel_size - stride) // 2


def smoothed_exp_margin(beta: float, margins: torch.Tensor) -> torch.Tensor:
    """
    Computes a beta-smoothed version of the margin using the exponential cost function.
    :param beta:
    :param margins:
    :return:
    """
    with torch.no_grad():
        beta_margins = -beta * margins
        max_beta_margin = beta_margins.max()
        sum_exp = torch.exp(beta_margins - max_beta_margin).sum()
        log_sum_exp = max_beta_margin + torch.log(sum_exp)
        smoothed_margin = (np.log(len(margins)) / beta) - (log_sum_exp / beta)
    return smoothed_margin


def smoothed_logistic_margin(beta: float, margins: torch.Tensor) -> torch.Tensor:
    """
    Computes a beta-smoothed version of the margin using the logistic cost function.
    :param beta:
    :param margins:
    :return:
    """
    with torch.no_grad():
        beta_margins = -beta * margins
        smoothed_margin = -1. / beta * np.log((torch.log(1 + torch.exp(beta_margins))).sum() / len(margins))
    return smoothed_margin


def get_standard_mf_lr_exponents(L: int):
    """
    Return the list c of layer-wise learning rate exponents for standard Mean Field models, i.e. c[0] = -1, c[l] = -2
    for l in [1, L-1], and c[L] = -1.
    :param L: int, number of hidden layers, i.e. the total number of layers is L+1.
    :return: list of (L+1) learning rate exponents
    """
    c = [-1] + [-2 for _ in range(1, L - 1)] + [-1]
    return c
