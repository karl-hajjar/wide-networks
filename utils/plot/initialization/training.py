import torch
import numpy as np
from copy import deepcopy

from pytorch.optimizers import OPT_DICT
from .network import define_networks


def compute_dists_to_init(nets, nets_init, average=True, norm=1):
    if norm != 'inf':
        if not (isinstance(norm, float, ) or isinstance(norm, int)):
            raise ValueError("Norm {} not implemented".format(norm))
        else:
            if norm == 0.:
                raise ValueError("'norm' argument cannot be 0")

    dist_to_init = dict()
    for m in nets.keys():
        nets_m = nets[m]
        nets_init_m = nets_init[m]
        dist_to_init[m] = []

        for i, net in enumerate(nets_m):
            net_init = nets_init_m[i]

            with torch.no_grad():
                param = torch.nn.utils.parameters_to_vector(net.parameters())
                param_init = torch.nn.utils.parameters_to_vector(net_init.parameters())
                diff = param - param_init
                if norm == 'inf':
                    dist = diff.abs().max().item()
                else:
                    if average:
                        diff = diff / diff.numel()
                    dist = torch.linalg.norm(diff, ord=norm).item()

            dist_to_init[m].append(dist)
    return dist_to_init


def compute_losses(nets, x, y):
    test_loss = dict()
    for m in nets.keys():
        test_loss[m] = []
        for net in nets[m]:
            net.eval()
            with torch.no_grad():
                y_pred = net(x)
                loss = torch.clamp(((y_pred - y).abs()), min=0, max=50).mean().item()
                test_loss[m].append(loss)

    return test_loss


def generate_test_results(L, ms, d, opt, opt_params, init, init_params, n_trials, x, y, x_test, y_test, batch_size=256,
                          steps=int(1.0e3), bias=False, scaling='mf', scale_init=None, epochs=500, average=True,
                          norm=1, collect_grads=True):

    nets = define_networks(L, ms, d, init, init_params, n_trials, bias, scaling, scale_init)
    nets_init = {m: [deepcopy(net) for net in nets[m]] for m in ms}

    return collect_test_results(opt, opt_params, nets, nets_init, batch_size, x, y, x_test, y_test, steps, epochs,
                                average, norm, collect_grads)


def collect_test_results(opt, opt_params, nets, nets_init, batch_size, x, y, x_test, y_test, steps, epochs=500,
                         average=True, norm=1, collect_grads=True):
    n_steps = 0
    dists_to_init_per_step = [(n_steps, compute_dists_to_init(nets, nets_init))]
    test_losses_per_step = [(n_steps, compute_losses(nets, x_test, y_test))]

    grads_per_step = []  # stays empty if collect_grads is False

    optimizers = _define_optimizers(nets, opt, opt_params)

    n = len(x)
    e = 0
    while (n_steps < steps) and (e < epochs):
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        x_train = x[indexes, :]
        y_train = y[indexes]

        for i in range(n // batch_size + 1):
            if n_steps >= steps:
                break

            batch_input = x_train[i * batch_size: (i + 1) * batch_size, :]
            batch_labels = y_train[i * batch_size: (i + 1) * batch_size]

            # forward passes + gradients computation
            grads_per_step_dict = dict()  # not used if collect_grads is False
            for m in nets.keys():
                opts_m = optimizers[m]
                grads_per_step_dict[m] = []  # not used if collect_grads is False
                for j, net in enumerate(nets[m]):
                    opt_m_j = opts_m[j]
                    net.train()  # set back to training mode
                    opt_m_j.zero_grad()

                    batch_output = net(batch_input)
                    loss = torch.clamp((batch_output - batch_labels).abs(), min=0, max=50).mean()
                    loss.backward()
                    opt_m_j.step()
                    if collect_grads:
                        grads_per_step_dict[m].append(_collect_grads(net))

            grads_per_step.append((n_steps, grads_per_step_dict))
            n_steps += 1

        dists_to_init_per_step.append((n_steps, compute_dists_to_init(nets, nets_init, average, norm)))
        test_losses_per_step.append((n_steps, compute_losses(nets, x_test, y_test)))

        e += 1

    return dists_to_init_per_step, test_losses_per_step, grads_per_step


def _define_optimizers(nets, opt, opt_params):
    return {m: [OPT_DICT[opt](net.parameters(), **opt_params) for net in nets[m]] for m in nets.keys()}


def _collect_grads(net):
    layer_grads = dict()
    L = len(net.layers)
    # gradient are those over the previously seen batch
    for l in range(L - 1):  # handle all layers except output layer
        layer_grads[l] = net.layers[l].layer[0].weight.grad.abs().max().item()
    # handle output layer separately
    l = L - 1
    layer_grads[l] = net.layers[l].layer.weight.grad.abs().max().item()
    return layer_grads
