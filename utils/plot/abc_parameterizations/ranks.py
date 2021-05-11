import numpy as np
import pandas as pd
from copy import deepcopy
import math
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def plot_weights_ranks_vs_layer(key: str, ranks_dfs: list, tol, L, width, lr, batch_size, marker='o', y_scale=None):
    plt.title('Rank of the weight matrix {} vs layer, L={}, m={}, lr={}, batchsize={}'.format(key, L, width, lr,
                                                                                              batch_size))

    tol = str(tol)
    sns.lineplot(data=ranks_dfs[0].astype(float), x='layer', y=key, marker=marker, label='svd rank default')
    sns.lineplot(data=ranks_dfs[1].astype(float), x='layer', y=key, marker=marker, label='svd rank {}'.format(tol))
    g = sns.lineplot(data=ranks_dfs[2].astype(float), x='layer', y=key, marker=marker, label='squared tr rank',
                     linestyle='--')

    if y_scale == 'log':
        g.set(yscale="log")
    plt.ylabel('rank')


def plot_acts_ranks_vs_layer(key: str, ranks_dfs: list, tol, L, width, lr, batch_size, marker='o', y_scale=None):
    plt.title('Rank of the activation matrix {} vs layer, L={}, m={}, lr={}, batchsize={}'.format(key, L, width, lr,
                                                                                                  batch_size))

    tol = str(tol)
    sns.lineplot(data=ranks_dfs[0].astype(float), x='layer', y=key, marker=marker, label='svd rank default')
    sns.lineplot(data=ranks_dfs[1].astype(float), x='layer', y=key, marker=marker, label='svd rank {}'.format(tol))
    g = sns.lineplot(data=ranks_dfs[2].astype(float), x='layer', y=key, marker=marker, label='squared tr rank',
                     linestyle='--')

    if y_scale == 'log':
        g.set(yscale="log")
    plt.ylabel('rank')
