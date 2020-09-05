import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

from . import set_plot_options
# see https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/ on how
# to hide axis/label/ticks


def plot_cluster_centers2d(data, figsize=(10, 10), style='darkgrid', marker=None, color='k', s=None, show_grid=True,
                           show_ticks=True):
    set_plot_options(style)
    plt.figure(figsize=figsize)
    # keep only first two coordinates of each cluster center
    plt.scatter(data[:, 0], data[:, 1], marker=marker, c=color, s=s)
    plt.grid(show_grid)
    if not show_ticks:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def plot_cluster_centers3d(data, figsize=(10, 10), style=None, marker=None, color='k', show_grid=True,
                           show_ticks=True):
    d = data.shape[1]
    if d < 3:
        raise ValueError("data points in data must be of dimension at least 3 but were of dimension {}".format(d))
    set_plot_options(style)
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    # keep only first 3 coordinates of each cluster center
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker, c=color)
    ax.grid(show_grid)
    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def plot_random_data2d(data_points: np.array, cluster_centers: np.array, labels: np.array, figsize=(10, 10),
                       style='darkgrid', marker=None, color='k', s=None, show_grid=True, show_ticks=True):
    """
    Takes randomly generated data as input and plots it in 2d. The plot consists of cluster centers along with the data
    points themselves, colored in red for the data points with a positive label (+1) and blue for the data points with a
    negative label (-1).
    :param data_points:
    :param cluster_centers:
    :param labels:
    :param figsize:
    :param style:
    :param marker:
    :param color:
    :param s:
    :param show_grid:
    :param show_ticks:
    :return:
    """
    set_plot_options(style)
    plt.figure(figsize=figsize)

    positive_data_points = data_points[labels == 1, :]
    negative_data_points = data_points[labels == -1, :]

    # keep only first two coordinates of each vector when plotting
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker=marker, c=color, s=s)  # cluster centers
    # plot data points with label +1
    plt.scatter(positive_data_points[:, 0], positive_data_points[:, 1], marker='+', c='r', s=s)
    # plot data points with label +1
    plt.scatter(negative_data_points[:, 0], negative_data_points[:, 1], marker='_', c='b', s=s)

    plt.grid(show_grid)
    if not show_ticks:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def plot_random_data3d(data_points: np.array, cluster_centers: np.array, labels: np.array, figsize=(10, 10),
                       style='darkgrid', marker=None, color='k', show_grid=True, show_ticks=True):
    """
    Takes randomly generated data as input and plots it in 2d. The plot consists of cluster centers along with the data
    points themselves, colored in red for the data points with a positive label (+1) and blue for the data points with a
    negative label (-1).
    :param data_points:
    :param cluster_centers:
    :param labels:
    :param figsize:
    :param style:
    :param marker:
    :param color:
    :param s:
    :param show_grid:
    :param show_ticks:
    :return:
    """
    d = data_points.shape[1]
    if d < 3:
        raise ValueError("data points in data must be of dimension at least 3 but were of dimension {}".format(d))
    set_plot_options(style)
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    positive_data_points = data_points[labels == 1, :]
    negative_data_points = data_points[labels == -1, :]

    # keep only first 3 coordinates of each vector
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker=marker, c=color)  # cluster centers
    # plot data points with label +1
    plt.scatter(positive_data_points[:, 0], positive_data_points[:, 1], positive_data_points[:, 2], marker='+', c='r')
    # plot data points with label +1
    plt.scatter(negative_data_points[:, 0], negative_data_points[:, 1], negative_data_points[:, 2], marker='_', c='b')

    ax.grid(show_grid)
    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
