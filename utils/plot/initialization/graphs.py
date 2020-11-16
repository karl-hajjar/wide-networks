import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils.plot import set_plot_options  # see __init__ file
from utils.plot.initialization.network import define_networks
from utils.plot.initialization.dataframes import *

figures_dir = '/Users/karlhajjar/Documents/projects/wide-networks/figures/initialization/'


def plot_average_over_trials(df, x, y, hue=None, linestyle=None, units=None, ax=None, figsize=(10, 10), style='darkgrid', title=None):
    set_plot_options(style)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=linestyle, units=units, palette="deep")
    plt.title(title)


def _get_template_figname(init, scaling, scale_init, L, prefix=''):
    if (prefix is not None) and (prefix != ''):
        prefix = prefix + '-'
    else:
        prefix = ''
    template_figname = prefix + '{}-{}-{}-{{y}}_vs_{{x}}'.format(init, scaling, L)
    if scale_init is not None:
        template_figname += '-' + scale_init
    template_figname += '.png'
    return template_figname


def generate_outputs_plots(figsize, L, ms, d, x, init, init_params, n_trials, scaling, bias=False, scale_init=None,
                           save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)

    nets = define_networks(L, ms, d, init, init_params, n_trials, bias=bias, scaling=scaling, scale_init=scale_init)
    layers_outputs_df = get_layers_outputs_df(nets, inputs=x)

    title = 'Max absolute value of layers outputs vs m in the {} scaling'.format(scaling)
    plot_average_over_trials(layers_outputs_df, x='m', y='max absolute output', hue='layer',
                             figsize=figsize, title=title)

    if save:
        figname = template_figname.format(y='all_layers_outputs', x='m')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()
    print('\n\n')

    for l in range(L):
        layer_outputs_l = layers_outputs_df.loc[layers_outputs_df.layer == l + 1, :]
        title = 'Max absolute value of layer {:,} outputs vs m in the {} scaling'.format(l + 1, scaling)
        plot_average_over_trials(layer_outputs_l, x='m', y='max absolute output', figsize=figsize,
                                 title=title)

        if save:
            figname = template_figname.format(y='layer_{:,}_output'.format(l + 1), x='m')
            plt.savefig(os.path.join(figures_dir, figname))

        plt.show()


def generate_gradients_plots(figsize, L, ms, d, x, init, init_params, n_trials, scaling, bias=False, scale_init=None,
                             save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)

    nets = define_networks(L, ms, d, init, init_params, n_trials, bias=bias, scaling=scaling, scale_init=scale_init)
    layers_gradients_df = get_layers_gradients_df(nets, inputs=x)

    title = 'Max absolute value of layers gradients vs m in the {} scaling'.format(scaling)
    plot_average_over_trials(layers_gradients_df, x='m', y='max absolute derivative', hue='layer',
                             figsize=figsize, title=title)

    if save:
        figname = template_figname.format(y='all_layers_gradients', x='m')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()
    print('\n\n')

    for l in range(L):
        layer_gradients_l = layers_gradients_df.loc[layers_gradients_df.layer == l + 1, :]
        title = 'Max absolute value of layer {:,} gradient vs m in the {} scaling'.format(l + 1, scaling)
        plot_average_over_trials(layer_gradients_l, x='m', y='max absolute derivative', figsize=figsize,
                                 title=title)

        if save:
            figname = template_figname.format(y='layer_{:,}_gradient'.format(l + 1), x='m')
            plt.savefig(os.path.join(figures_dir, figname))

        plt.show()


def generate_rescaled_gradients_plots(figsize, L, ms, d, x, init, init_params, n_trials, scaling, bias=False,
                                      scale_init=None, save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)

    nets = define_networks(L, ms, d, init, init_params, n_trials, bias=bias, scaling=scaling, scale_init=scale_init)
    layers_rescaled_gradients_df = get_layers_rescaled_gradients_df(nets, inputs=x)

    title = 'Max absolute value of rescaled layers gradients vs m in the {} scaling'.format(scaling)
    plot_average_over_trials(layers_rescaled_gradients_df, x='m', y='max absolute rescaled derivative', hue='layer',
                             figsize=figsize, title=title)

    if save:
        figname = template_figname.format(y='all_layers_rescaled_gradients', x='m')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()
    print('\n\n')

    for l in range(L):
        layer_rescaled_gradients_l = layers_rescaled_gradients_df.loc[layers_rescaled_gradients_df.layer == l + 1, :]
        title = 'Max absolute value of layer {:,} rescaled gradient vs m in the {} scaling'.format(l + 1, scaling)
        plot_average_over_trials(layer_rescaled_gradients_l, x='m', y='max absolute rescaled derivative',
                                 figsize=figsize, title=title)

        if save:
            figname = template_figname.format(y='layer_{:,}_rescaled_gradient'.format(l + 1), x='m')
            plt.savefig(os.path.join(figures_dir, figname))

        plt.show()


def plot_gradients_m_vs_step(df, m, L, figsize, init, scaling, scale_init=None, save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)
    title = 'Max absolute value of layers gradients vs steps in the {} scaling with m={:,}'.format(scaling, m)
    plot_average_over_trials(df, x='step', y='max absolute derivative', hue='layer', figsize=figsize, title=title)

    if save:
        figname = template_figname.format(y='net_{:,}_all_layers_gradients'.format(m), x='steps')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()


def plot_gradients_vs_step(df, L, figsize, init, scaling, scale_init=None, save=True, prefix=''):
    for m in df.m.unique():
        df_m = df.loc[df.m == m, :]
        plot_gradients_m_vs_step(df_m, m, L, figsize, init, scaling, scale_init, save, prefix)


def plot_test_losses_vs_step(df, L, figsize, init, scaling, scale_init=None, save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)
    title = 'Test loss vs steps in the {} scaling'.format(scaling)
    plot_average_over_trials(df, x='step', y='test loss', figsize=figsize, hue='m', title=title)

    if save:
        figname = template_figname.format(y='all_test_losses', x='steps')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()


def plot_dists_to_init_vs_step(df, L, figsize, init, scaling, scale_init=None, save=True, prefix=''):
    template_figname = _get_template_figname(init, scaling, scale_init, L, prefix)
    title = 'Distance to init vs steps in the {} scaling'.format(scaling)
    plot_average_over_trials(df, x='step', y='distance to init', figsize=figsize, hue='m', title=title)

    if save:
        figname = template_figname.format(y='all_dists_to_init', x='steps')
        plt.savefig(os.path.join(figures_dir, figname))

    plt.show()

