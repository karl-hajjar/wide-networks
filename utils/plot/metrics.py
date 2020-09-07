import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from . import set_plot_options


def plot_metric_vs_other(x_metric, y_metric, ax=None, figsize=(10, 10), style='darkgrid', marker='+', color='r', linewidth=None,
                         title=None, xlabel=None, ylabel=None, legend=True):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    assert len(y_metric) == len(x_metric), "x_metric and y_metric must have same length but x_metric had length {:,} " \
                                           "and y_metric had length {:,}".format(len(x_metric), len(y_metric))
    # scatter y_metric values for all trials for each value of x_metric
    for i in range(len(y_metric)):
        label = None
        if i == 0:
            label = 'randomized trials'
        ys = y_metric[i]
        ax.scatter([x_metric[i]] * len(ys), ys, marker=marker, c=color, label=label)

    # line plot trial means of y_metric vs x_metric
    y_means = np.mean(y_metric, axis=1)
    ax.plot(x_metric, y_means, linewidth=linewidth, label='mean')

    if title is None:
        title = 'y metric vs x metric'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend:
        plt.legend()


def plot_metric_vs_other_mutiple_params(x_metric, y_metric, params, param_name, title, xlabel, ylabel, figsize=(16, 10),
                                        style='darkgrid', marker='+', color='r', linewidth=None):
    assert len(y_metric) == len(x_metric) == len(params), \
        "x_metric, y_metric and params must have same length but had respective lengths {:,}, {:,} and {:,}".\
            format(len(x_metric), len(y_metric), len(params))

    n_max = 12
    max_width = 4
    n_plots = min(len(params), n_max)
    width = min(n_plots, max_width)
    height = math.ceil(n_plots / width)

    fig, axs = plt.subplots(height, width, figsize=figsize)
    fig.suptitle(title, fontsize=18)

    for i in range(height):
        for j in range(width):
            index = i * width + j
            if index < len(params):  # otherwise there is no dqtq to plot
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
                title = '{}={}'.format(param_name, params[index])
                plot_metric_vs_other(x_metric[index], y_metric[index], ax, figsize=None, style=style, marker=marker,
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
