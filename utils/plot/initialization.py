import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import torch

from . import set_plot_options  # see __init__ file


def plot_average_over_trials(df, x, y, hue=None, ax=None, figsize=(10, 10), style='darkgrid', title=None):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    sns.lineplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)


def get_all_layers_df(nets, n=100, compute_outputs=False):
    dfs = []
    columns = ['m', 'l', 'weights average norm']
    if compute_outputs:
        columns.append('output')
    for m in nets.keys():
        for net in nets[m]:
            for l in range(len(net.layers)):
                weights_l = net.layers[l].layer[1].weight.data.detach()
                average_weight_vector_l = weights_l.mean(axis=0)
                df = pd.DataFrame(columns=columns)


def get_layers_outputs_df(nets, inputs):
    columns = ['m', 'l', 'average output absolute mean']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for m in nets.keys():
        for net in nets[m]:
            a = inputs
            for l in range(len(net.layers)):
                a = net.layers[l].forward(a).detach()
                # compute for each neuron the average activation over samples, then take the mean over neurons of the
                # absolute values of those average activations
                df.loc[i, columns] = [m, l, a.mean(dim=0).abs().mean().item()]
                i += 1
    return df.astype(float)
