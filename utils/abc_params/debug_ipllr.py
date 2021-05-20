from copy import deepcopy
import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils.nn import squared_trace_rank, frob_spec_rank


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


def get_W0_dict(model_0, normalize_first=True):
    L = model_0.n_layers - 1
    layer_scales = model_0.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        W0 = {1: layer_scales[0] * model_0.input_layer.weight.data.detach()}
        for i, l in enumerate(range(2, L + 1)):
            layer = getattr(model_0.intermediate_layers, intermediate_layer_keys[i])
            W0[l] = layer_scales[l - 1] * layer.weight.data.detach()

        W0[L + 1] = layer_scales[L] * model_0.output_layer.weight.data.detach()

        b0 = layer_scales[0] * model_0.input_layer.bias.data.detach()

        if normalize_first:
            W0[1] = W0[1] / math.sqrt(model_0.d + 1)
            b0 = b0 / math.sqrt(model_0.d + 1)

    return W0, b0


def get_W0_b0_dict(model_0, normalize_first=True):
    L = model_0.n_layers - 1
    layer_scales = model_0.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        W0 = {1: layer_scales[0] * model_0.input_layer.weight.data.detach()}
        b0 = {1: model_0.input_layer.bias.data.detach()}
        for i, l in enumerate(range(2, L + 1)):
            layer = getattr(model_0.intermediate_layers, intermediate_layer_keys[i])
            W0[l] = layer_scales[l - 1] * layer.weight.data.detach()
            b0[l] = layer.bias.data.detach()

        W0[L + 1] = layer_scales[L] * model_0.output_layer.weight.data.detach()
        b0[L + 1] = model_0.output_layer.bias.data.detach()

        if normalize_first:
            W0[1] = W0[1] / math.sqrt(model_0.d + 1)
            b0[1] = b0[1] / math.sqrt(model_0.d + 1)

    return W0, b0


def get_Delta_W1_dict(model_0, model_1, normalize_first=True):
    L = model_0.n_layers - 1
    layer_scales = model_0.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        Delta_W_1 = {1: layer_scales[0] * (model_1.input_layer.weight.data.detach() - 
                                           model_0.input_layer.weight.data.detach())}
        for i, l in enumerate(range(2, L + 1)):
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            layer_0 = getattr(model_0.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_1[l] = layer_scales[l - 1] * (layer_1.weight.data.detach() -
                                                  layer_0.weight.data.detach())

        Delta_W_1[L + 1] = layer_scales[L] * (model_1.output_layer.weight.data.detach() -
                                              model_0.output_layer.weight.data.detach())

        Delta_b_1 = layer_scales[0] * (model_1.input_layer.bias.data.detach() - 
                                       model_0.input_layer.bias.data.detach())

    if normalize_first:
        Delta_W_1[1] = Delta_W_1[1] / math.sqrt(model_1.d + 1)
        Delta_b_1 = Delta_b_1 / math.sqrt(model_1.d + 1)
    
    return Delta_W_1, Delta_b_1


def get_Delta_W1_b1_dict(model_0, model_1, normalize_first=True):
    L = model_0.n_layers - 1
    layer_scales = model_0.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        Delta_W_1 = {1: layer_scales[0] * (model_1.input_layer.weight.data.detach() -
                                           model_0.input_layer.weight.data.detach())}
        Delta_b_1 = {1: model_1.input_layer.bias.data.detach() -
                        model_0.input_layer.bias.data.detach()}
        for i, l in enumerate(range(2, L + 1)):
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            layer_0 = getattr(model_0.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_1[l] = layer_scales[l - 1] * (layer_1.weight.data.detach() -
                                                  layer_0.weight.data.detach())
            Delta_b_1[l] = layer_1.bias.data.detach() - layer_0.bias.data.detach()

        Delta_W_1[L + 1] = layer_scales[L] * (model_1.output_layer.weight.data.detach() -
                                              model_0.output_layer.weight.data.detach())
        Delta_b_1[L + 1] = model_1.output_layer.bias.data.detach() - model_0.output_layer.bias.data.detach()

    if normalize_first:
        Delta_W_1[1] = Delta_W_1[1] / math.sqrt(model_1.d + 1)
        Delta_b_1[1] = Delta_b_1[1] / math.sqrt(model_1.d + 1)

    return Delta_W_1, Delta_b_1


def get_Delta_W2_dict(model_1, model_2, normalize_first=True):
    L = model_1.n_layers - 1
    layer_scales = model_1.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        Delta_W_2 = {1: layer_scales[0] * (model_2.input_layer.weight.data.detach() -
                                           model_1.input_layer.weight.data.detach())}
        for i, l in enumerate(range(2, L + 1)):
            layer_2 = getattr(model_2.intermediate_layers, intermediate_layer_keys[i])
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_2[l] = layer_scales[l - 1] * (layer_2.weight.data.detach() -
                                                  layer_1.weight.data.detach())

        Delta_W_2[L + 1] = layer_scales[L] * (model_2.output_layer.weight.data.detach() -
                                              model_1.output_layer.weight.data.detach())

        Delta_b_2 = layer_scales[0] * (model_2.input_layer.bias.data.detach() -
                                       model_1.input_layer.bias.data.detach())

    if normalize_first:
        Delta_W_2[1] = Delta_W_2[1] / math.sqrt(model_2.d + 1)
        Delta_b_2 = Delta_b_2 / math.sqrt(model_2.d + 1)

    return Delta_W_2, Delta_b_2


def get_Delta_W2_b2_dict(model_1, model_2, normalize_first=True):
    L = model_1.n_layers - 1
    layer_scales = model_1.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]
    with torch.no_grad():
        Delta_W_2 = {1: layer_scales[0] * (model_2.input_layer.weight.data.detach() -
                                           model_1.input_layer.weight.data.detach())}
        Delta_b_2 = {1: model_2.input_layer.bias.data.detach() -
                        model_1.input_layer.bias.data.detach()}
        for i, l in enumerate(range(2, L + 1)):
            layer_2 = getattr(model_2.intermediate_layers, intermediate_layer_keys[i])
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_2[l] = layer_scales[l - 1] * (layer_2.weight.data.detach() -
                                                  layer_1.weight.data.detach())
            Delta_b_2[l] = layer_2.bias.data.detach() - layer_1.bias.data.detach()

        Delta_W_2[L + 1] = layer_scales[L] * (model_2.output_layer.weight.data.detach() -
                                              model_1.output_layer.weight.data.detach())
        Delta_b_2[L + 1] = model_2.output_layer.bias.data.detach() - model_1.output_layer.bias.data.detach()

    if normalize_first:
        Delta_W_2[1] = Delta_W_2[1] / math.sqrt(model_2.d + 1)
        Delta_b_2[1] = Delta_b_2[1] / math.sqrt(model_2.d + 1)

    return Delta_W_2, Delta_b_2


def get_contributions_1(x, model_1, W0, b0, Delta_W_1, Delta_b_1, normalize_first=True):
    L = model_1.n_layers - 1
    layer_scales = model_1.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]

    bias = model_1.bias
    scale_bias = model_1.scale_bias
    
    with torch.no_grad():
        x1 = {0: x}
        h0 = {1: F.linear(x, W0[1], b0)}
        delta_h_1 = {1: F.linear(x, Delta_W_1[1], Delta_b_1)}
        h1 = {1: layer_scales[0] * model_1.input_layer.forward(x)}
        if normalize_first:
            h1[1] = h1[1] / math.sqrt(model_1.d + 1)
        x1[1] = model_1.activation(h1[1])

        torch.testing.assert_allclose(h0[1] + delta_h_1[1], h1[1], rtol=1e-4, atol=1e-4)

        for i, l in enumerate(range(2, L + 1)):
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            x = x1[l - 1]

            h0[l] = F.linear(x, W0[l])
            delta_h_1[l] = F.linear(x, Delta_W_1[l])

            h1[l] = layer_scales[l - 1] * layer_1.forward(x)
            x1[l] = model_1.activation(h1[l])

            torch.testing.assert_allclose(h0[l] + delta_h_1[l], h1[l], rtol=1e-4, atol=1e-4)

        x = x1[L]
        h0[L + 1] = F.linear(x, W0[L + 1])
        delta_h_1[L + 1] = F.linear(x, Delta_W_1[L + 1])
        h1[L + 1] = layer_scales[L] * model_1.output_layer.forward(x)
        x1[L + 1] = model_1.activation(h1[L + 1])

        torch.testing.assert_allclose(h0[L + 1] + delta_h_1[L + 1], h1[L + 1], rtol=1e-4, atol=1e-4)

    return h0, delta_h_1, h1, x1


def get_contributions_1_bias(x, model_1, W0, b0, Delta_W_1, Delta_b_1, normalize_first=True):
    L = model_1.n_layers - 1
    layer_scales = model_1.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]

    with torch.no_grad():
        x1 = {0: x}
        h0 = {1: F.linear(x, W0[1], b0[1])}
        delta_h_1 = {1: F.linear(x, Delta_W_1[1], Delta_b_1[1])}
        h1 = {1: model_1.input_layer.forward(layer_scales[0] * x)}
        if normalize_first:
            h1[1] = h1[1] / math.sqrt(model_1.d + 1)
        x1[1] = model_1.activation(h1[1])

        torch.testing.assert_allclose(h0[1] + delta_h_1[1], h1[1], rtol=1e-4, atol=1e-4)

        for i, l in enumerate(range(2, L + 1)):
            layer_1 = getattr(model_1.intermediate_layers, intermediate_layer_keys[i])
            x = x1[l - 1]

            h0[l] = F.linear(x, W0[l], b0[l])
            delta_h_1[l] = F.linear(x, Delta_W_1[l], Delta_b_1[l])

            h1[l] = layer_1.forward(layer_scales[l - 1] * x)
            x1[l] = model_1.activation(h1[l])

            torch.testing.assert_allclose(h0[l] + delta_h_1[l], h1[l], rtol=1e-4, atol=1e-4)

        x = x1[L]
        h0[L + 1] = F.linear(x, W0[L + 1], b0[L+1])
        delta_h_1[L + 1] = F.linear(x, Delta_W_1[L + 1], Delta_b_1[L+1])
        h1[L + 1] = model_1.output_layer.forward(layer_scales[L] * x)
        x1[L + 1] = model_1.activation(h1[L + 1])

        torch.testing.assert_allclose(h0[L + 1] + delta_h_1[L + 1], h1[L + 1], rtol=1e-4, atol=1e-4)

    return h0, delta_h_1, h1, x1


def get_svd_ranks_weights(W0, Delta_W_1, Delta_W_2, L, tol=None):
    columns = ['layer', 'W0', 'Delta_W_1', 'Delta_W_2', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            if tol is None:
                df.loc[l, columns] = [l,
                                      torch.matrix_rank(W0[l]).item(),
                                      torch.matrix_rank(Delta_W_1[l]).item(),
                                      torch.matrix_rank(Delta_W_2[l]).item(),
                                      min(W0[l].shape[0], W0[l].shape[1])]
            else:
                df.loc[l, columns] = [l,
                                      torch.matrix_rank(W0[l], tol=tol).item(),
                                      torch.matrix_rank(Delta_W_1[l], tol=tol).item(),
                                      torch.matrix_rank(Delta_W_2[l], tol=tol).item(),
                                      min(W0[l].shape[0], W0[l].shape[1])]

    return df


def get_square_trace_ranks_weights(W0, Delta_W_1, Delta_W_2, L):
    columns = ['layer', 'W0', 'Delta_W_1', 'Delta_W_2', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            df.loc[l, columns] = [l,
                                  squared_trace_rank(W0[l]),
                                  squared_trace_rank(Delta_W_1[l]),
                                  squared_trace_rank(Delta_W_2[l]),
                                  min(W0[l].shape[0], W0[l].shape[1])]

    return df


def get_frob_spec_ranks_weights(W0, Delta_W_1, Delta_W_2, L):
    columns = ['layer', 'W0', 'Delta_W_1', 'Delta_W_2', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            df.loc[l, columns] = [l,
                                  frob_spec_rank(W0[l]),
                                  frob_spec_rank(Delta_W_1[l]),
                                  frob_spec_rank(Delta_W_2[l]),
                                  min(W0[l].shape[0], W0[l].shape[1])]

    return df


def get_svd_ranks_acts(h0, delta_h_1, h1, x1, L, tol=None):
    columns = ['layer', 'h0', 'delta_h_1', 'h1', 'x1', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            if tol is None:
                df.loc[l, columns] = [l,
                                      torch.matrix_rank(h0[l]).item(),
                                      torch.matrix_rank(delta_h_1[l]).item(),
                                      torch.matrix_rank(h1[l]).item(),
                                      torch.matrix_rank(x1[l]).item(),
                                      min(h0[l].shape[0], h0[l].shape[1])]
            else:
                df.loc[l, columns] = [l,
                                      torch.matrix_rank(h0[l], tol=tol).item(),
                                      torch.matrix_rank(delta_h_1[l], tol=tol).item(),
                                      torch.matrix_rank(h1[l], tol=tol).item(),
                                      torch.matrix_rank(x1[l], tol=tol).item(),
                                      min(h0[l].shape[0], h0[l].shape[1])]

    return df


def get_square_trace_ranks_acts(h0, delta_h_1, h1, x1, L):
    columns = ['layer', 'h0', 'delta_h_1', 'h1', 'x1', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            df.loc[l, columns] = [l,
                                  squared_trace_rank(h0[l]),
                                  squared_trace_rank(delta_h_1[l]),
                                  squared_trace_rank(h1[l]),
                                  squared_trace_rank(x1[l]),
                                  min(h0[l].shape[0], h0[l].shape[1])]

    return df


def get_frob_spec_ranks_acts(h0, delta_h_1, h1, x1, L):
    columns = ['layer', 'h0', 'delta_h_1', 'h1', 'x1', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            df.loc[l, columns] = [l,
                                  frob_spec_rank(h0[l]),
                                  frob_spec_rank(delta_h_1[l]),
                                  frob_spec_rank(h1[l]),
                                  frob_spec_rank(x1[l]),
                                  min(h0[l].shape[0], h0[l].shape[1])]

    return df


def get_max_acts_diversity(h0, delta_h_1, h1, L):
    columns = ['layer', 'h0', 'delta_h_1', 'h1', 'max']
    df = pd.DataFrame(columns=columns, index=range(1, L + 2))
    df.index.name = 'layer'

    with torch.no_grad():
        for l in df.index:
            maxes = dict()

            _, maxes['h0'] = torch.max(h0[l], dim=1)
            _, maxes['delta_h_1'] = torch.max(delta_h_1[l], dim=1)
            _, maxes['h1'] = torch.max(h1[l], dim=1)

            df.loc[l, ['h0', 'delta_h_1', 'h1']] = [maxes[key].unique().numel() for key in ['h0', 'delta_h_1', 'h1']]
            df.loc[l, 'layer'] = l
            df.loc[l, 'max'] = h0[l].shape[0]

    return df


def get_concatenated_ranks_df(ranks_dfs: list) -> pd.DataFrame:
    return pd.concat(ranks_dfs, axis=0, ignore_index=True)


def get_avg_ranks_dfs(ranks_df: pd.DataFrame) -> pd.DataFrame:
    avg_weight_ranks_df = ranks_df.astype(float).groupby(by='layer', as_index=True).mean()
    avg_weight_ranks_df = avg_weight_ranks_df.astype(int)
    return avg_weight_ranks_df
