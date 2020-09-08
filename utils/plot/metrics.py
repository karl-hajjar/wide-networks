import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from . import set_plot_options  # see __init__ file


def plot_metric_vs_param(metric, param, ax=None, figsize=(10, 10), style='darkgrid', marker='+', color='r',
                         linewidth=None, title=None, xlabel=None, ylabel=None, legend=True):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    assert len(param) == len(metric), "metric and param must have same length but metric had length {:,} " \
                                      "and param had length {:,}".format(len(metric), len(param))
    # scatter y_metric values for all trials for each value of x_metric
    for i in range(len(metric)):
        label = None
        if i == 0:
            label = 'randomized trials'
        ys = metric[i]
        ax.scatter([param[i]] * len(ys), ys, marker=marker, c=color, label=label)

    # line plot trial means of y_metric vs x_metric
    y_means = np.mean(metric, axis=1)
    ax.plot(param, y_means, linewidth=linewidth, label='mean')

    if title is None:
        title = 'y metric vs x metric'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend:
        plt.legend()


def plot_metric_vs_param_mutiple(metric, param, other_params, other_param_name, title, xlabel, ylabel, figsize=(16, 10),
                                 style='darkgrid', marker='+', color='r', linewidth=None):
    assert len(param) == len(metric) == len(other_params), \
        "metric, param and other_params must have same length but had respective lengths {:,}, {:,} and {:,}". \
            format(len(metric), len(param), len(other_params))

    n_max = 12
    max_width = 4
    n_plots = min(len(other_params), n_max)
    width = min(n_plots, max_width)
    height = math.ceil(n_plots / width)

    fig, axs = plt.subplots(height, width, figsize=figsize)
    fig.suptitle(title, fontsize=18)

    for i in range(height):
        for j in range(width):
            index = i * width + j
            if index < len(other_params):  # otherwise there is no dqtq to plot
                legend = False
                if height == 1:  # if only one row than axs has only one dimension
                    ax = axs[j]
                else:
                    ax = axs[i, j]
                if i == height - 1:  # add xlabel only for the last row
                    xlabel_ax = xlabel
                else:
                    xlabel_ax = None
                if j == 0:  # add ylabel only for the first column
                    ylabel_ax = ylabel
                    if i == 0:  # only one legend is enough for the whole plot
                        legend = True
                else:
                    ylabel_ax = None
                title = '{}={}'.format(other_param_name, other_params[index])
                plot_metric_vs_param(metric[index], param[index], ax, figsize=None, style=style, marker=marker,
                                     color=color, linewidth=linewidth, title=title, xlabel=xlabel_ax, ylabel=ylabel_ax,
                                     legend=legend)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')


def plot_metric_vs_step(metric, steps, ax=None, figsize=(10, 10), style='darkgrid', title=None, xlabel='steps',
                        ylabel='metric'):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    assert len(metric) == len(steps), "metric and steps must have same length but metric had length {:,} " \
                                      "and steps had length {:,}".format(len(metric), len(steps))

    ax.plot(steps, metric)

    if title is None:
        title = 'y metric vs x metric'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_metric_vs_param_from_results(other_params_dict, results, metric_key, param_key, param_values, other_param_key,
                                      n_trials=5, mode='test', ax=None, figsize=(10, 10), style='darkgrid', marker='+',
                                      color='r', linewidth=None, legend=True):
    metric_name = ' '.join(metric_key.split('_'))
    param_to_string = dict()
    for key, value in other_params_dict.items():
        if key != param_key:
            param_to_string[key] = '{}={}'.format(key, value)
        else:
            param_to_string[key] = key + '={}'
    base_exp_name = param_to_string['bsize'] + '_' + param_to_string['ntrain'] + '_' + param_to_string['d'] + '_' + \
                    param_to_string['m']

    metric = []
    params_keep = []
    if mode == 'test':
        index = 0
    elif mode in ['training', 'validation']:
        index = -1  # if training or validation, get the last value as reference for convergence
    else:
        raise ValueError("mode argument must be one of ['training', 'validation', 'test'] but was {}".format(mode))
    for param in param_values:
        exp_name = base_exp_name.format(param)
        try:
            metric.append([results[exp_name][i][mode][index][metric_key] for i in range(n_trials)])
            params_keep.append(param)
        except Exception as e:
            print('Metric retrieval failed for {}={} : {}'.format(param_key, param, e))
    if metric_key == 'beta':
        title = '{} vs {} with {}={:,}'.format(metric_name, param_key, other_param_key,
                                               other_params_dict[other_param_key])
    else:
        title = '{} {} vs {} with {}={:,}'.format(mode, metric_name, param_key, other_param_key,
                                                  other_params_dict[other_param_key])
    xlabel = param_key
    ylabel = metric_name
    plot_metric_vs_param(metric, params_keep, ax=ax, figsize=figsize, style=style, marker=marker, color=color,
                         linewidth=linewidth, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
