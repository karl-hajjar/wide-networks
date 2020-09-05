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
