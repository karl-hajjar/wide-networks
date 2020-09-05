import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from . import set_plot_options


def plot_metric_vs_other(x_metric, y_metric, ax=None, figsize=(10, 10), style='darkgrid', marker='+', color='r', linewidth=None,
                         title=None, xlabel='x metric', ylabel='y metric'):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    assert len(y_metric) == len(x_metric), "x_metric and y_metric must have same length but metric 1 had length {:,} " \
                                           "and metric2 had length {:,}".format(len(x_metric), len(y_metric))
    # scatter y_metric values for all trials for each value of x_metric
    for i in range(len(y_metric)):
        ys = y_metric[i]
        ax.scatter([x_metric[i]] * len(ys), ys, marker=marker, c=color)

    # line plot trial means of y_metric vs x_metric
    y_means = np.mean(y_metric, axis=1)
    ax.plot(x_metric, y_means, linewidth=linewidth)

    if title is None:
        title = 'y metric vs x metric'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


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
    fig.suptitle(title, fontsize=16)

    for i in range(height):
        for j in range(width):
            if i == height - 1:
                axs[i, j].set_xlabel(xlabel)
            if j == 0:
                axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_title('{}={}'.format(param_name, params[i * width + j]))
            #axs[i, j].plot(, y1, color='b')
    ## fig 1
    # plt.subplot(2,2,1) # plt.subplot(number of lines, number of columns, index of the current subplot)
    # axs[0, 0].set_xlabel('x input')
    # axs[0, 0].set_ylabel('f(x) output')
    # axs[0, 0].set_ylim(0, 100)
    # axs[0, 0].set_title('Squared function of input')
    # axs[0, 0].plot(x, y1, color='b')
    #
    # ## fig 2
    # axs[0, 1].set_xlabel('x input')
    # axs[0, 1].set_ylabel('f(x) output')
    # axs[0, 1].set_ylim(0, 100)
    # axs[0, 1].set_title('Squared function of input')
    # axs[0, 1].scatter(x, y2, color='r', marker='x')
    #
    # ## fig 3
    # axs[1, 0].set_xlabel('x input')
    # axs[1, 0].set_ylabel('f(x) output')
    # axs[1, 0].set_ylim(0, 100)
    # axs[1, 0].set_title('Squared function of input')
    # axs[1, 0].plot(x, y3, color='g')
    #
    # ## fig 4
    # axs[1, 1].set_xlabel('x input')
    # axs[1, 1].set_ylabel('f(x) output')
    # axs[1, 1].set_ylim(0, 100)
    # axs[1, 1].set_title('Squared function of input')
    # axs[1, 1].plot(x, y4, color='orange', label='alpha = {}'.format(alpha4))
    # axs[1, 1].scatter(x, y5, c='purple', label='alpha = {}'.format(alpha5))
    # axs[1, 1].legend()

    ## if you want to remove the tick values on the x-axis (for instance)
    plt.setp([a.get_xticklabels() for a in axs[0, :]], visible=False)

    ## 2 following lines are optional, they are just here to ajust the space between plots and main title. Commenting them
    # out will still produce a nice plot
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9)

    # display all figures
    plt.show()
