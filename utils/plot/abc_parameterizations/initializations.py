import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def generate_init_outputs(model_class, n_trials, widths, config, x=None, bias=False, **args):
    if x is None:
        x = 2 * torch.rand(config.architecture["input_size"], requires_grad=False) - 1
    norm_2_x = torch.norm(x, 2).item()
    if np.abs(norm_2_x - 1) > 1e-6:
        x = x / norm_2_x  # normalized x

    results_df = pd.DataFrame(columns=['m', 'output', 'abs output', 'squared output'], dtype=float)
    idx = 0

    for width in widths:
        config = deepcopy(config)
        config.architecture["width"] = width
        config.architecture["bias"] = False

        for _ in range(n_trials):
            model = model_class(config, width=None, **args)
            model.train()
            model.initialize_params(config.initializer)
            if not bias:
                with torch.no_grad():
                    model.input_layer.bias.data.fill_(0.)  # reset bias to zero
            model.eval()
            with torch.no_grad():
                output = model.forward(x).detach()[0].item()
                results_df.loc[idx, :] = [width, output, np.abs(output), output ** 2]
                idx += 1

    return results_df


def plot_init_outputs_vs_m(fig_path, model_name, model_class, n_trials, widths, config, x=None, figsize=(12, 6),
                           bias=False, x_scale='log', y_scale='log', save=True, show=True, marker='o', **args):
    results_df = generate_init_outputs(model_class, n_trials, widths, config, x, bias, **args)
    L = config.architecture['n_layers'] - 1
    d = config.architecture["input_size"]
    var = 1 / (d + 1)
    exponent = _get_exponent_from_name(L, activation=config.activation.name, name=model_name)

    plt.figure(figsize=figsize)
    plt.title('{} output at initialization vs m with L={} hidden layers'.format(model_name, L))

    sns.lineplot(data=results_df, x='m', y='abs output', marker=marker, label='abs output')
    plt.plot(widths, math.sqrt(var) * (np.array(widths, dtype=float) ** (-exponent / 2)), marker=marker,
             label='m^(-{}/2)'.format(exponent))

    g = sns.lineplot(data=results_df, x='m', y='squared output', marker=marker, label='squared output')
    plt.plot(widths, var * (np.array(widths, dtype=float) ** (-exponent)), marker=marker,
             label='m^(-{})'.format(exponent))

    if y_scale == 'log':
        g.set(yscale="log")
        plt.yscale('log')
        plt.ylabel('output (log scale)')
    else:
        plt.ylabel('output')

    if x_scale == 'log':
        g.set(xscale="log")
        plt.xscale('log')
        plt.xlabel('width (log scale)')
    else:
        plt.xlabel('width')

    plt.legend()
    if save:
        plt.savefig(fig_path)
    if show:
        plt.show()

    return results_df


def plot_init_outputs_dist(fig_path, model_name, model_class, n_trials, widths, config, x=None,
                           figsize=(12, 6), save=True, show=True, *args):
    results_df = generate_init_outputs(model_class, n_trials, widths, config, x, *args)

    d = config.architecture["input_size"]
    var = 1 / (d + 1)
    n = math.ceil(np.sqrt(len(widths)))
    fig, axs = plt.subplots(n, n)
    fig.suptitle('{} output distribution at initialization'.format(model_name))

    xs = np.arange(start=-1.0, stop=1.0, step=0.02)
    ys = (1 / np.sqrt(2 * math.pi * var)) * np.exp(-(xs ** 2) / (2 * var))

    for k, width in enumerate(widths):
        i, j = divmod(k, n)
        ax = axs[i, j]
        outputs = results_df.loc[results_df.m == width, 'output'].values
        sns.histplot(outputs, kde=True, stat='density', ax=ax)
        ax.figure.set_size_inches(figsize[0], figsize[1])
        ax.plot(xs, ys, label='std Gaussian', c='r')
        ax.legend()
        ax.set_xlabel('output')
        ax.set_ylabel('output density')
        ax.set_title('m = {:,}'.format(width))

    plt.legend()
    if save:
        plt.savefig(fig_path)
    if show:
        plt.show()


def _get_exponent_from_name(L: int, activation: str, name: str):
    if 'ntk' in name.lower():
        return 0
    elif 'mup' in name.lower():
        return 1
    elif 'ip' in name.lower():
        if 'relu_' in activation:
            q = int(activation.split('_')[1])
            return q * L
        return L