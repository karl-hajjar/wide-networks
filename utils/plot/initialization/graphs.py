import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils.plot import set_plot_options  # see __init__ file


def plot_average_over_trials(df, x, y, hue=None, ax=None, figsize=(10, 10), style='darkgrid', title=None):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    sns.lineplot(data=df, x=x, y=y, hue=hue, palette="deep")
    plt.title(title)


def get_all_layers_df(nets, n=100, compute_outputs=False):
    dfs = []
    columns = ['m', 'layer', 'weights average norm']
    if compute_outputs:
        columns.append('output')
    for m in nets.keys():
        for net in nets[m]:
            for l in range(len(net.layers)):
                weights_l = net.layers[l].layer[1].weight.data.detach()
                average_weight_vector_l = weights_l.mean(axis=0)
                df = pd.DataFrame(columns=columns)


def get_layers_outputs_df(nets, inputs):
    columns = ['m', 'layer', 'max absolute output']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for m in nets.keys():
        for net in nets[m]:
            a = inputs
            for l in range(len(net.layers)):
                a = net.layers[l].forward(a).detach()
                # df.loc[i, columns] = [m, l, a.abs().mean().item()]
                df.loc[i, columns] = [m, l+1, a.abs().max().item()]
                i += 1

    df[['m', 'layer']] = df[['m', 'layer']].astype(int)
    df['max absolute output'] = df['max absolute output'].astype(float)
    return df


def get_layers_gradients_df(nets, inputs):
    columns = ['m', 'layer', 'max absolute derivative']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for m in nets.keys():
        for net in nets[m]:
            net.train()
            outputs_mean = net(inputs).mean()
            outputs_mean.backward()
            L = len(net.layers)
            for l in range(L-1):  # handle all layers except output layer
                grad = net.layers[l].layer[0].weight.grad
                # note that grad holds the average gradient of the output over all samples
                # df.loc[i, columns] = [m, l, grad.abs().mean().item()]
                df.loc[i, columns] = [m, l+1, grad.abs().max().item()]
                i += 1
            # handle output layer separately
            l = L - 1
            grad = net.layers[l].layer.weight.grad
            df.loc[i, columns] = [m, l+1, grad.abs().max().item()]
            i += 1

    df[['m', 'layer']] = df[['m', 'layer']].astype(int)
    df['max absolute derivative'] = df['max absolute derivative'].astype(float)
    return df


def get_layers_rescaled_gradients_df(nets, inputs):
    columns = ['m', 'layer', 'max absolute rescaled derivative']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for m in nets.keys():
        for net in nets[m]:
            net.train()
            outputs_mean = net(inputs).mean()
            outputs_mean.backward()
            L = len(net.layers)

            # define the scaling to use for the gradients depending on the scaling used for the network
            if net.scaling == 'mf':
                scale = m
            elif net.scaling == 'ntk':
                scale = np.sqrt(m)
            else:
                raise ValueError("scaling {} is not implemented.".format(scaling))

            # handle the case first layer (l=0)
            l = 0
            grad = scale * net.layers[0].layer[0].weight.grad  # rescale gradient by scale
            df.loc[i, columns] = [m, l+1, grad.abs().max().item()]
            i += 1

            # handle intermediate layers
            for l in range(1, L-1):  # handle all layers except output layer
                grad = (scale ** 2) * net.layers[l].layer[0].weight.grad  # rescale gradient by scale^2
                # note that grad holds the average gradient of the output over all samples
                # df.loc[i, columns] = [m, l, grad.abs().mean().item()]
                df.loc[i, columns] = [m, l+1, grad.abs().max().item()]
                i += 1

            # handle output layer
            l = L - 1
            grad = scale * net.layers[l].layer.weight.grad  # rescale gradient by scale
            df.loc[i, columns] = [m, l+1, grad.abs().max().item()]
            i += 1

    df[['m', 'layer']] = df[['m', 'layer']].astype(int)
    df['max absolute rescaled derivative'] = df['max absolute rescaled derivative'].astype(float)
    return df
