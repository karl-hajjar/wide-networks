import numpy as np
import pandas as pd
from copy import deepcopy
import math
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def plot_losses(data_ip, data_muP, key, L, width, lr, batch_size, mode, marker='o', normalize_first=True):
    data_df_ip = pd.DataFrame(columns=['step', key], dtype=float)
    idx = 0
    for data_ip_ in data_ip:
        for i, loss in enumerate(data_ip_):
            data_df_ip.loc[idx, ['step', key]] = [i + 1, loss]
            idx += 1

    if data_muP is not None:
        data_df_muP = pd.DataFrame(columns=['step', key], dtype=float)
        idx = 0
        for losses_muP_ in data_muP:
            for i, loss in enumerate(losses_muP_):
                data_df_muP.loc[idx, ['step', key]] = [i + 1, loss]
                idx += 1

    plt.title('{} {} vs steps, L={}, m={}, lr={}, batchsize={}, renorm 1st layer={}'.format(mode, key, L, width, lr,
                                                                                            batch_size,
                                                                                            normalize_first))
    sns.lineplot(data=data_df_ip, x='step', y=key, marker=marker, label='IPLLR')
    if data_muP is not None:
        sns.lineplot(data=data_df_muP, x='step', y=key, marker=marker, label='muP')


def plot_output_scale(ip_dfs, muP_dfs, layer, key, L, width, lr, batch_size, mode, marker='o', y_scale=None):
    ip_df = pd.concat(ip_dfs, axis=0, ignore_index=True)
    outputs_df_ip = ip_df.loc[ip_df.layer == layer, :]

    if muP_dfs is not None:
        muP_df = pd.concat(muP_dfs, axis=0, ignore_index=True)
        outputs_df_muP = muP_df.loc[muP_df.layer == layer, :]

    plt.title('{} {} at layer {} vs steps, L={}, m={}, lr={},  batchsize={}'.format(mode, key, layer, L, width, lr,
                                                                                    batch_size))
    sns.lineplot(data=outputs_df_ip, x='step', y=key, marker=marker, label='IPLLR')
    if muP_dfs is not None:
        g = sns.lineplot(data=outputs_df_muP, x='step', y=key, marker=marker, label='muP')

        if y_scale == 'log':
            g.set(yscale="log")
