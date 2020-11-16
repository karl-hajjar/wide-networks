import torch
import pandas as pd
import numpy as np


def get_all_layers_df(nets, n=100, compute_outputs=False):
    dfs = []
    columns = ['m', 'layer', 'weights average norm']
    if compute_outputs:
        columns.append('output')
    with torch.no_grad():
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
    with torch.no_grad():
        for m in nets.keys():
            for net in nets[m]:
                a = inputs
                for l in range(len(net.layers)):
                    a = net.layers[l].forward(a).detach()
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
                raise ValueError("scaling {} is not implemented.".format(net.scaling))

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


def get_all_grads_df(grads):
    columns = ['m', 'step', 'layer', 'max absolute derivative']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for step, grads_dict in grads:
        for m, grads_dict_m in grads_dict.items():
            for layer_grads in grads_dict_m:
                for l, max_abs_grad in layer_grads.items():
                    df.loc[i, columns] = [m, step, l+1, max_abs_grad]
                    i += 1

    df[['m', 'step', 'layer']] = df[['m', 'step', 'layer']].astype(int)
    df['max absolute derivative'] = df['max absolute derivative'].astype(float)

    return df


def get_all_test_losses_df(test_losses):
    columns = ['m', 'step', 'test loss']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for step, losses_dict in test_losses:
        for m in losses_dict.keys():
            for loss in losses_dict[m]:
                df.loc[i, columns] = [m, step, loss]
                i += 1

    df[['m', 'step']] = df[['m', 'step']].astype(int)
    df['test loss'] = df['test loss'].astype(float)

    return df


def get_all_dist_to_init_df(dists_to_init):
    columns = ['m', 'step', 'distance to init']
    df = pd.DataFrame(columns=columns, dtype=float)
    i = 0
    for step, dist_dict in dists_to_init:
        for m in dist_dict.keys():
            for dist in dist_dict[m]:
                df.loc[i, columns] = [m, step, dist]
                i += 1

    df[['m', 'step']] = df[['m', 'step']].astype(int)
    df['distance to init'] = df['distance to init'].astype(float)

    return df
