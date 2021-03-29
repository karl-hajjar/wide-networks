import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import logging
from scipy.interpolate import lagrange
from collections.abc import Iterable
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def generate_1d_data(n_samples: int = 10):
    # uniform samples in [-1, 1]
    xs = torch.linspace(start=-1.0, end=1.0, steps=n_samples)
    ys = 2 * torch.rand(size=(len(xs),)) - 1
    return xs, ys


def fit_model(model, x, y, n_epochs=1, init_bias=True):
    if not init_bias:
        with torch.no_grad():
            model.input_layer.bias.data.fill_(0.)
    model.train()
    for i in range(n_epochs):  # full batch gradients
        model.optimizer.zero_grad()
        y_hat = model.forward(x)
        loss = model.loss(y_hat, y)
        loss.backward()

        model.optimizer.step()
        if hasattr(model, 'scheduler') and (model.scheduler is not None):
            model.scheduler.step()

    if not init_bias:
        with torch.no_grad():
            model.input_layer.bias.data.fill_(0.)
    model.eval()


def plot_model_vs_other(model1, model2, xs, ys, num=100, figsize=(12, 6), c='k'):
    x = np.linspace(start=-1.0, stop=1.0, num=num)
    plt.scatter(xs, ys, marker='o', c=c, label='training points')


def plot_model(model, xs, ys, label, x=None,  num=100, c='k', scatter=False):
    if x is None:
        x = np.linspace(start=-1.0, stop=1.0, num=num)
    if scatter:
        plt.scatter(xs, ys, marker='o', c=c, label='training points')

    if isinstance(model, Iterable):  # plot functions with std
        y_df = pd.DataFrame(columns=['x', 'y'], dtype=float)
        idx = 0
        for model_ in model:
            model_.eval()
            with torch.no_grad():
                batch_x = torch.unsqueeze(torch.Tensor(x), 1)
                y = model_.forward(batch_x).numpy()
            for i, x_ in enumerate(x):
                y_df.loc[idx, ['x', 'y']] = [x_, y[i]]
                idx += 1

        sns.lineplot(data=y_df, x='x', y='y', label=label)

    else:  # plot only a single model
        model.eval()
        with torch.no_grad():
            batch_x = torch.unsqueeze(torch.Tensor(x), 1)
            y = model.forward(batch_x).numpy()
        plt.plot(x, y, label=label)


def plot_training(models, xs, ys, label, num=100, c='k', n_epochs=10, init_bias=True, secs=0.3, fig=None,
                  figsize=(12, 6)):
    plt.ion()
    plot_model(models, xs, ys, label, x=None, num=num, c=c, scatter=True)
    plt.draw()
    plt.cla()
    time.sleep(secs)

    batch_xs = torch.unsqueeze(xs, 1)
    batch_ys = torch.unsqueeze(ys, 1)
    for i in range(n_epochs):
        for model in models:
            fit_model(model, batch_xs, batch_ys, n_epochs=1, init_bias=init_bias)
        plot_model(models, xs, ys, label, x=None, num=num, c=c, scatter=True)
        plt.draw()
        time.sleep(secs)

