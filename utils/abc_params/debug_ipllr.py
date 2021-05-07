import os
from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F

from utils.tools import read_yaml
from pytorch.configs.base import BaseConfig
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
from pytorch.models.abc_params.fully_connected.muP import FCmuP
from pytorch.models.abc_params.fully_connected.ntk import FCNTK
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from utils.dataset.mnist import load_data


def train_model_one_step(model, x, y, normalize_first=True, verbose=True):
    model.train()
    batch_size = len(x)

    # set gradients to 0
    model.optimizer.zero_grad()

    # outputs at initialization
    y_hat = model.forward(x, normalize_first=normalize_first)
    y_hat.retain_grad()
    loss = model.loss(y_hat, y)

    # gradients at initialization
    loss.backward()

    if verbose:
        print('input abs mean in training: ', x.abs().mean().item())
        print('loss derivatives for model:', batch_size * y_hat.grad)
        print('average training loss for model1 :', np.mean(loss.item()))
        print('')

    # first weight update
    model.optimizer.step()

    if hasattr(model, "scheduler") and model.scheduler is not None:
        model.scheduler.step()


def train_model_one_step_with_loss(model, x, y, normalize_first=True, verbose=True):
    model.train()
    batch_size = len(x)

    # set gradients to 0
    model.optimizer.zero_grad()

    # outputs
    y_hat = model.forward(x, normalize_first=normalize_first)
    y_hat.retain_grad()
    loss = model.loss(y_hat, y)

    # gradients
    loss.backward()

    if verbose:
        print('input abs mean in training: ', x.abs().mean().item())
        print('loss derivatives for model:', batch_size * y_hat.grad)
        print('average training loss for model :', np.mean(loss.item()))
        print('')

    # first weight update
    model.optimizer.step()

    if hasattr(model, "scheduler") and model.scheduler is not None:
        model.scheduler.step()

    chi = (batch_size * y_hat.grad).abs().mean().item()
    return loss.item(), chi


def compute_contributions_with_previous(model, model_init, model_previous, batches, step, normalize_first=True):
    model.eval()
    model_init.eval()
    model_previous.eval()

    contributions_df = pd.DataFrame(columns=['layer', 'h_init', 'Delta_h', 'delta_h', 'h', 'step'],
                                    dtype=float)

    idx = 0

    L = model.n_layers - 1
    with torch.no_grad():
        losses = []
        for i, batch in enumerate(batches):
            x, y = batch
            with torch.no_grad():
                x_0 = x.clone().detach()

            # input layer
            Delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                        model_init.input_layer.weight.data)
            Delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                        model_init.input_layer.bias.data)

            delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                        model_previous.input_layer.weight.data)
            delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                        model_previous.input_layer.bias.data)

            init_contrib = model_init.layer_scales[0] * model_init.input_layer.forward(x)
            Delta_h = F.linear(x, Delta_W, Delta_b)
            delta_h = F.linear(x, delta_W, delta_b)
            total_contrib = model.layer_scales[0] * model.input_layer.forward(x)

            if normalize_first:
                init_contrib = init_contrib / math.sqrt(model_init.d + 1)
                Delta_h = Delta_h / math.sqrt(model.d + 1)
                delta_h = delta_h / math.sqrt(model.d + 1)
                total_contrib = total_contrib / math.sqrt(model.d + 1)

            torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                          rtol=1e-3, atol=1e-0)

            contributions_df.loc[idx, ['layer', 'h_init', 'Delta_h', 'delta_h', 'h', 'step']] = \
                [1, init_contrib.abs().mean().item(), Delta_h.abs().mean().item(),
                 delta_h.abs().mean().item(), total_contrib.abs().mean().item(), step]
            idx += 1

            x = model.activation(total_contrib)

            # intermediate layer grads
            for l in range(2, L + 1):
                layer_key = "layer_{:,}_intermediate".format(l)
                layer = getattr(model.intermediate_layers, layer_key)
                init_layer = getattr(model_init.intermediate_layers, layer_key)
                previous_layer = getattr(model_previous.intermediate_layers, layer_key)

                Delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)

                delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - previous_layer.weight.data)

                init_contrib = model_init.layer_scales[l - 1] * init_layer.forward(x)

                Delta_h = F.linear(x, Delta_W)
                delta_h = F.linear(x, delta_W)
                total_contrib = model.layer_scales[l - 1] * layer.forward(x)

                torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                              rtol=1e-2, atol=1e-2)

                contributions_df.loc[idx, ['layer', 'h_init', 'Delta_h', 'delta_h', 'h', 'step']] = \
                    [l, init_contrib.abs().mean().item(), Delta_h.abs().mean().item(),
                     delta_h.abs().mean().item(), total_contrib.abs().mean().item(), step]
                idx += 1

                x = model.activation(total_contrib)

            # output layer
            Delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                        model_init.output_layer.weight.data)

            delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                        model_previous.output_layer.weight.data)

            init_contrib = model_init.layer_scales[L] * model_init.output_layer.forward(x)

            Delta_h = F.linear(x, Delta_W)
            delta_h = F.linear(x, delta_W)
            total_contrib = model.layer_scales[L] * model.output_layer.forward(x)

            torch.testing.assert_allclose(init_contrib + Delta_h, total_contrib,
                                          rtol=1e-2, atol=1e-2)

            contributions_df.loc[idx, ['layer', 'h_init', 'Delta_h', 'delta_h', 'h', 'step']] = \
                [L + 1, init_contrib.abs().mean().item(), Delta_h.abs().mean().item(),
                 delta_h.abs().mean().item(), total_contrib.abs().mean().item(), step]
            idx += 1

            y_hat_debug = total_contrib
            y_hat = model.forward(x_0, normalize_first=normalize_first)

            torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

            losses.append(model.loss(y_hat, y).item())

    return contributions_df


def compute_contributions_with_step_1(model_name, model, model_0, model_1, model_previous, batches,
                                      normalize_first=True):
    model.eval()
    model_0.eval()
    model_1.eval()

    contributions_df = pd.DataFrame(columns=['model', 'layer', 'h_1', 'Delta_h_1', 'Delta_h',
                                             'delta_h', 'h', 'id'])
    contributions_df.loc[:, ['layer', 'h_1', 'Delta_h_1', 'Delta_h', 'delta_h', 'h', 'id']] = \
        contributions_df.loc[:, ['layer', 'h_1', 'Delta_h_1', 'Delta_h', 'delta_h', 'h', 'id']].astype(float)

    idx = 0

    L = model.n_layers - 1
    with torch.no_grad():
        losses = []
        for i, batch in enumerate(batches):
            x, y = batch
            with torch.no_grad():
                x_0 = x.clone().detach()

            # input layer
            Delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                        model_0.input_layer.weight.data)
            Delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                        model_0.input_layer.bias.data)

            Delta_W_1 = (model.width ** (-model.a[0])) * (model_1.input_layer.weight.data -
                                                          model_0.input_layer.weight.data)
            Delta_b_1 = (model.width ** (-model.a[0])) * (model_1.input_layer.bias.data -
                                                          model_0.input_layer.bias.data)

            delta_W = (model.width ** (-model.a[0])) * (model.input_layer.weight.data -
                                                        model_previous.input_layer.weight.data)
            delta_b = (model.width ** (-model.a[0])) * (model.input_layer.bias.data -
                                                        model_previous.input_layer.bias.data)

            h_1 = model_1.layer_scales[0] * model_1.input_layer.forward(x)
            Delta_h_1 = F.linear(x, Delta_W_1, Delta_b_1)
            Delta_h = F.linear(x, Delta_W, Delta_b)
            delta_h = F.linear(x, delta_W, delta_b)
            total_contrib = model.layer_scales[0] * model.input_layer.forward(x)

            if normalize_first:
                h_1 = h_1 / math.sqrt(model_1.d + 1)
                Delta_h_1 = Delta_h_1 / math.sqrt(model.d + 1)
                Delta_h = Delta_h / math.sqrt(model.d + 1)
                delta_h = delta_h / math.sqrt(model.d + 1)
                total_contrib = total_contrib / math.sqrt(model.d + 1)

            contributions_df.loc[idx, ['model', 'layer', 'h_1', 'Delta_h_1', 'Delta_h', 'delta_h', 'h', 'id']] = \
                [model_name, 1, h_1.abs().mean().item(), Delta_h_1.abs().mean().item(), Delta_h.abs().mean().item(),
                 delta_h.abs().mean().item(), total_contrib.abs().mean().item(), i]
            idx += 1

            x = model.activation(total_contrib)

            # intermediate layer grads
            for l in range(2, L + 1):
                layer_key = "layer_{:,}_intermediate".format(l)
                layer = getattr(model.intermediate_layers, layer_key)
                init_layer = getattr(model_0.intermediate_layers, layer_key)
                layer_1 = getattr(model_1.intermediate_layers, layer_key)
                previous_layer = getattr(model_previous.intermediate_layers, layer_key)

                Delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)
                Delta_W_1 = (model_1.width ** (-model_1.a[l - 1])) * (layer_1.weight.data - init_layer.weight.data)

                delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - previous_layer.weight.data)

                h_1 = model_1.layer_scales[l - 1] * layer_1.forward(x)

                Delta_h_1 = F.linear(x, Delta_W_1)
                Delta_h = F.linear(x, Delta_W)
                delta_h = F.linear(x, delta_W)
                total_contrib = model.layer_scales[l - 1] * layer.forward(x)

                contributions_df.loc[idx, ['model', 'layer', 'h_1', 'Delta_h_1', 'Delta_h', 'delta_h', 'h', 'id']] = \
                    [model_name, l, h_1.abs().mean().item(), Delta_h_1.abs().mean().item(), Delta_h.abs().mean().item(),
                     delta_h.abs().mean().item(), total_contrib.abs().mean().item(), i]
                idx += 1

                x = model.activation(total_contrib)

            # output layer
            Delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                        model_0.output_layer.weight.data)
            Delta_W_1 = (model_1.width ** (-model.a[L])) * (model_1.output_layer.weight.data -
                                                            model_0.output_layer.weight.data)
            delta_W = (model.width ** (-model.a[L])) * (model.output_layer.weight.data -
                                                        model_previous.output_layer.weight.data)

            h_1 = model_1.layer_scales[L] * model_1.output_layer.forward(x)
            Delta_h_1 = F.linear(x, Delta_W_1)
            Delta_h = F.linear(x, Delta_W)
            delta_h = F.linear(x, delta_W)
            total_contrib = model.layer_scales[L] * model.output_layer.forward(x)

            contributions_df.loc[idx, ['model', 'layer', 'h_1', 'Delta_h_1', 'Delta_h', 'delta_h', 'h', 'id']] = \
                [model_name, L + 1, h_1.abs().mean().item(), Delta_h_1.abs().mean().item(), Delta_h.abs().mean().item(),
                 delta_h.abs().mean().item(), total_contrib.abs().mean().item(), i]
            idx += 1

            y_hat_debug = total_contrib
            y_hat = model.forward(x_0, normalize_first=normalize_first)

            torch.testing.assert_allclose(y_hat_debug, y_hat, rtol=1e-5, atol=1e-5)

            losses.append(model.loss(y_hat, y).item())

    print('average validation loss for {} : {}'.format(model_name, np.mean(losses)))
    return contributions_df


def collect_scales(model, model_init, batches, eval_batch, n_steps, normalize_first=True, verbose=False, eval=False):
    training_results = []
    eval_results = []
    training_losses = []
    training_chis = []
    model_previous = deepcopy(model)  # copy previous parameters
    for i in range(n_steps):
        batch = batches[i % len(batches)]

        # compute contributions
        training_results.append(compute_contributions_with_previous(model, model_init, model_previous, [batch],
                                                                    i + 1, normalize_first=normalize_first))  # training

        if eval:  # evaluation
            eval_results.append(compute_contributions_with_previous(model, model_init, model_previous, [eval_batch],
                                                                    i + 1, normalize_first=normalize_first))

        # train model for one step
        x, y = batch
        loss, chi = train_model_one_step_with_loss(model, x, y, normalize_first=normalize_first, verbose=verbose)
        training_losses.append(loss)
        training_chis.append(chi)

    if eval:
        eval_results = pd.concat(eval_results, axis=0, ignore_index=True)
    else:
        eval_results = []

    return training_losses, training_chis, pd.concat(training_results, axis=0, ignore_index=True), eval_results


def collect_training_losses(model, batches, n_steps, normalize_first=True, verbose=False):
    training_losses = []
    training_chis = []
    for i in range(n_steps):
        batch = batches[i % len(batches)]

        # train model for one step
        x, y = batch
        loss, chi = train_model_one_step_with_loss(model, x, y, normalize_first=normalize_first, verbose=verbose)
        training_losses.append(loss)
        training_chis.append(chi)

    return training_losses, training_chis
