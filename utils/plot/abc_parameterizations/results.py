import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from utils.tools import load_pickle


def split_epoch_step_results(results, pop_keys=('lrs', 'all_losses')):
    popped_results = dict()
    for key in pop_keys:
        if key in results['training'][0].keys():
            popped_results[key] = []
            for k in range(len(results['training'])):
                popped_results[key].extend(results['training'][k].pop(key))

    return results, popped_results


def get_trial_results(Ls, widths, n_trials, exp_dir, base_exp, activation, lr, batch_size, bias, n_warmup_steps=None):
    """
    Loads the results from multiple trials of the same experiment defined by the choice of L, width, activation, lr,
    batch size, bias. Here we allow loading results for multiple experiments with different values for L and width.
    :param Ls:
    :param widths:
    :param n_trials:
    :param exp_dir:
    :param base_exp:
    :param activation:
    :param lr:
    :param batch_size:
    :param bias:
    :return:
    """
    res_dict = dict()
    model_config = 'activation={}_lr={}_batchsize={}_bias={}'.format(activation, lr, batch_size, bias)
    if n_warmup_steps is not None:
        model_config += '_warmup={}'.format(n_warmup_steps)
    for L in Ls:
        res_dict[L] = dict()
        for width in widths:
            res = []
            for idx in range(1, n_trials + 1):
                results_path = os.path.join(
                    exp_dir,
                    base_exp,
                    'L={}_m={}'.format(L, width),
                    model_config,
                    'trial_{}'.format(idx),
                    'results.pickle')
                # print(results_path)
                if not os.path.exists(results_path):
                    logging.warning('results for trial {:,} with L={:,} and m={:,} was not found at {}'\
                                    .format(idx, L, width, results_path))
                else:
                    res.append(load_pickle(results_path, single=True))
            res_dict[L][width] = res
    return res_dict


def get_epoch_step_results_from_trials(results, pop_keys=('lrs', 'all_losses')):
    """
    Split the results obtained from get_trial_results into epoch and step results. Multiple values of L and width are
    thus expected in the keys of results.
    :param results:
    :param pop_keys:
    :return:
    """
    epoch_res_dict = dict()
    step_res_dict = dict()
    for L in results.keys():
        epoch_res_dict[L] = dict()
        step_res_dict[L] = dict()
        for width in results[L].keys():
            epoch_res = []
            step_res = []
            for res in results[L][width]:
                epoch_r, step_r = split_epoch_step_results(res, pop_keys=pop_keys)
                epoch_res.append(epoch_r)
                step_res.append(step_r)

            epoch_res_dict[L][width] = epoch_res
            step_res_dict[L][width] = step_res

    return epoch_res_dict, step_res_dict


def plot_metric_vs_time(L, results: [list, dict], metric: str, time: str, save_path: str, metric_name: str = None,
                        mode: str = None, figsize=(16,10), marker='o', y_min=None, y_max=None, logscale=False,
                        show=True, save=True):
    """
    Plots a certain metric either vs # optimization steps or vs # epochs. `time` can only be 'step' if `mode` is
    'training'.
    :param results: dictionary containing the results of a single trial or list of dictionaries containing the results
    of multiple trials.
    :param metric: str, the name of the metric to plot.
    :param metric_name: str, the name to use in the legend of the plot for the mertic.
    :param time: str, either 'epoch' or 'step'.
    :param mode: str, 'training', 'validation', or None.
    :param y_min: the minimum value of y, default None
    :param y_max: the maximum value of y, default None
    :return:

    """
    if time == 'step':
        if mode == 'validation':
            logging.warning("`mode` was 'validation' even if `time` was 'step'. `mode` will be set to 'training' by "
                            "default.")
        mode = 'training'
    elif time == 'epoch':
        if mode not in ['training', 'validation']:
            raise ValueError("If `time` is 'epoch', `mode` must be one of 'training' or 'validation' but was {}".\
                             format(time))
    else:
        raise ValueError("`time` must be one of 'step' or 'epoch' but was {}".format(time))

    if isinstance(results, dict):
        results = [results]

    if metric_name is None:
        metric_name = metric

    if time == 'epoch':
        ys = [[r[metric] for r in res[mode]] for res in results]
    else:
        marker = None
        ys = [res[metric] for res in results]

    plt.figure(figsize=figsize)
    plt.title('{} {} vs # {}s with L={:,}'.format(mode, metric_name, time, L))
    for i, y in enumerate(ys):
        plt.plot(np.arange(1, len(y) + 1), y, marker=marker, label='trial {}'.format(i+1))

    if len(ys) > 1:
        plt.legend()

    plt.xlabel(time)

    plt.ylim(y_min, y_max)

    if logscale:
        plt.yscale('log')
        plt.ylabel('{} (log scale)'.format(metric_name))
    else:
        plt.ylabel('{}'.format(metric_name))

    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def plot_metric_vs_time_std(results: list, metric: str, time: str, ax=None, metric_name: str = None,
                            mode: str = None, label=None, marker='o', loc=None, legend_title=None):
    """
    Plots a certain metric either vs # optimization steps or vs # epochs averaged over multiple trials. `time` can
    only be 'step' if `mode` is 'training'.
    :param width: the width used to produce the results
    :param results: list of dictionaries containing the results of multiple trials.
    :param metric: str, the name of the metric to plot.
    :param metric_name: str, the name to use in the legend of the plot for the mertic.
    :param time: str, either 'epoch' or 'step'.
    :param mode: str, 'training', 'validation', or None.
    :return:

    """
    if time == 'step':
        if mode == 'validation':
            logging.warning("`mode` was 'validation' even if `time` was 'step'. `mode` will be set to 'training' by "
                            "default.")
        mode = 'training'
    elif time == 'epoch':
        if mode not in ['training', 'validation']:
            raise ValueError("If `time` is 'epoch', `mode` must be one of 'training' or 'validation' but was {}".\
                             format(time))
    else:
        raise ValueError("`time` must be one of 'step' or 'epoch' but was {}".format(time))

    if metric_name is None:
        metric_name = metric

    if ax is None:
        ax = plt.gca()

    y_df = pd.DataFrame(columns=[time, metric_name], dtype=float)
    idx = 0

    if time == 'epoch':
        for res in results:
            for i, r in enumerate(res[mode]):
                y_df.loc[idx, [time, metric_name]] = [i + 1, r[metric]]
                idx += 1
        # ys = [[r[metric] for r in res[mode]] for res in results]
    else:
        marker = None
        for res in results:
            for i, r in enumerate(res[metric]):
                y_df.loc[idx, [time, metric_name]] = [i + 1, r]
                idx += 1
        # ys = [res[metric] for res in results]

    sns.lineplot(data=y_df, x=time, y=metric_name, ax=ax, marker=marker, label=label)


def plot_metric_vs_time_std_widths(L, results: dict, metric: str, time: str, ax=None, metric_name: str = None,
                                   mode: str = None, marker='o', y_min=None, y_max=None):
    """
    Plots a certain metric either vs # optimization steps or vs # epochs for multiple widths. `time` can only be 'step'
    if `mode` is 'training'.
    :param results: dictionary containing the results for different widths.
    :param metric: str, the name of the metric to plot.
    :param metric_name: str, the name to use in the legend of the plot for the mertic.
    :param time: str, either 'epoch' or 'step'.
    :param mode: str, 'training', 'validation', or None.
    :param y_min: the minimum value of y, default None
    :param y_max: the maximum value of y, default None
    :return:
    """
    if time == 'step':
        if mode == 'validation':
            logging.warning("`mode` was 'validation' even if `time` was 'step'. `mode` will be set to 'training' by "
                            "default.")
        mode = 'training'
    elif time == 'epoch':
        if mode not in ['training', 'validation']:
            raise ValueError("If `time` is 'epoch', `mode` must be one of 'training' or 'validation' but was {}".\
                             format(time))
    else:
        raise ValueError("`time` must be one of 'step' or 'epoch' but was {}".format(time))

    if ax is None:
        ax = plt.gca()

    if metric_name is None:
        metric_name = metric

    if L in results.keys():
        results = results[L]

    for width in results.keys():
        plot_metric_vs_time_std(width, results[width], metric, time, ax, metric_name, mode, marker)

    ax.set_ylim(y_min, y_max)


def plot_metric_vs_time_std_L(fig_path: str, results: dict, metric: str, time: str, metric_name: str = None,
                              mode: str = None, figsize=(12,6), marker='o', y_min=None, y_max=None, save=True,
                              show=True):
    """
    Plots a certain metric either vs # optimization steps or vs # epochs for multiple widths. `time` can only be 'step'
    if `mode` is 'training'.
    :param results: dictionary containing the results for different number of layers and widths.
    :param metric: str, the name of the metric to plot.
    :param metric_name: str, the name to use in the legend of the plot for the mertic.
    :param time: str, either 'epoch' or 'step'.
    :param mode: str, 'training', 'validation', or None.
    :param y_min: the minimum value of y, default None
    :param y_max: the maximum value of y, default None
    :return:
    """
    if time == 'step':
        if mode == 'validation':
            logging.warning("`mode` was 'validation' even if `time` was 'step'. `mode` will be set to 'training' by "
                            "default.")
        mode = 'training'
    elif time == 'epoch':
        if mode not in ['training', 'validation']:
            raise ValueError("If `time` is 'epoch', `mode` must be one of 'training' or 'validation' but was {}".\
                             format(time))
    else:
        raise ValueError("`time` must be one of 'step' or 'epoch' but was {}".format(time))

    if metric_name is None:
        metric_name = metric

    n = math.ceil(np.sqrt(len(results)))
    fig, axs = plt.subplots(n, n)
    fig.suptitle('Standard IP {} {} vs # {}s'.format(mode, metric_name, time))
    plt.setp(axs, ylim=(y_min, y_max))

    for k, L in enumerate(results.keys()):
        i, j = divmod(k, n)
        ax = axs[i, j]
        ax.figure.set_size_inches(figsize[0], figsize[1])
        plot_metric_vs_time_std_widths(L, results[L], metric, time, ax, metric_name, mode, marker,
                                       y_min=None, y_max=None)
        ax.set_title('L = {:,}'.format(L))
        ax.legend()

    plt.legend()
    if save:
        plt.savefig(fig_path)
    if show:
        plt.show()


def set_figure_fontsizes(fontsize: [int, float] = 12, ax=None, legend_fontsize=None, ticks_fontsize=None,
                         labels_fontsize=None):
    if ax is None:
        ax = plt.gca()
    if legend_fontsize is None:
        legend_fontsize = fontsize
    if ticks_fontsize is None:
        ticks_fontsize = fontsize
    if labels_fontsize is None:
        labels_fontsize = fontsize

    # ax.legend(prop={'size': legend_fontsize})
    ax.tick_params(axis='x', labelsize=ticks_fontsize)
    ax.tick_params(axis='y', labelsize=ticks_fontsize)
    ax.xaxis.label.set_size(labels_fontsize)
    ax.yaxis.label.set_size(labels_fontsize)
