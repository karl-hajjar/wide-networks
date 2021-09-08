import torch
from copy import deepcopy
import logging
from torch.optim import SGD
import torch.nn.functional as F
from typing import Union
import math


class WarmupSwitchLR(torch.optim.lr_scheduler._LRScheduler):
    """
    A learning rate scheduler which simply switches from initial learning rate values to warm learning rate values after
    a certain number of warmup steps.
    """

    DEFAULT_CALIBRATED_INITIAL_BASE_LR = 1.0

    def __init__(self, optimizer, initial_lrs, warm_lrs, n_warmup_steps=1, base_lr=0.01, last_epoch=-1,
                 calibrate_base_lr=True, model=None, batches=None, default_calibration=False):
        self._check_lrs(optimizer, initial_lrs, warm_lrs)
        self._check_warmup_steps(n_warmup_steps)

        self.initial_lrs = initial_lrs
        self.warm_lrs = warm_lrs
        self.n_warmup_steps = n_warmup_steps
        self.base_lr = base_lr

        self.current_lrs = initial_lrs
        self._last_lrs = None

        # __init__ of _LRScheduler sets _step_count=0 and calls step() as a check which will increment _step_count to 1
        # as the initial value before any optimization takes place
        super().__init__(optimizer, last_epoch)  # sets self.optimizer = optimizer
        # also possible is to reset _step_count to 0 after the call to the parent __init__

        if calibrate_base_lr:
            if not default_calibration:
                if (model is None) or (batches is None):
                    raise ValueError("model and batches cannot be None when calibrating the initial base learning "
                                     "rate.")
                initial_base_lrs = self.calibrate_base_lr(model, batches)
                assert len(initial_base_lrs) == len(self.initial_lrs)
            else:
                initial_base_lrs = self.DEFAULT_CALIBRATED_INITIAL_BASE_LR

            self.initial_base_lrs = initial_base_lrs
        else:
            self.initial_base_lrs = self.base_lr

        self._set_param_group_lrs(self.initial_base_lrs)

    @staticmethod
    def _check_lrs(optimizer, initial_lrs, warm_lrs):
        n_param_groups = len(list(optimizer.param_groups))
        if (len(initial_lrs) != n_param_groups) or (len(warm_lrs) != n_param_groups):
            raise ValueError("There must be as many learning rates as there are parameter groups but there were {:,} "
                             "parameter groups but {:,} initial learning rates and {:,} warm learning rates".
                             format(n_param_groups, len(initial_lrs), len(warm_lrs)))

    @staticmethod
    def _check_warmup_steps(n_warmup_steps):
        if not isinstance(n_warmup_steps, int):
            raise TypeError("`n_warmup_steps` argument must be of type int")
        if n_warmup_steps <= 0:
            raise ValueError("`n_warmup_steps` argument must be > 0")

    # @staticmethod
    def calibrate_base_lr(self, model, batches):
        logging.info("Calibrating initial base learning rate")

        # use a (mock) copy of the model and optimizer so as not to modify the parameters of the object passed
        model_ = deepcopy(model)
        model_.train()

        base_lr_mock = 1.0
        # set mock SGD optimizer with base_lr = 1.0 so that no additional scaling factor appears
        param_groups = \
            [{'params': model_.input_layer.parameters(), 'lr': base_lr_mock * model_.lr_scales[0]}] + \
            [{'params': layer.parameters(), 'lr': base_lr_mock * model_.lr_scales[l+1]}
             for l, layer in enumerate(model_.intermediate_layers)] + \
            [{'params': model_.output_layer.parameters(), 'lr': base_lr_mock * model_.lr_scales[model_.n_layers - 1]}]
        optimizer = SGD(param_groups, lr=base_lr_mock)

        # remember initial weight values
        initial_model = deepcopy(model_)

        # take first step of optimization
        x, y = batches[0]
        y_hat = model_.forward(x)
        loss = model_.loss(y_hat, y)
        loss.backward()
        optimizer.step()

        # calibrate the lr using activations of second forward pass
        model_.eval()
        x, _ = batches[1]
        base_lrs = []
        with torch.no_grad():
            Delta_W_1 = (model_.width ** (-model_.a[0])) * (model_.input_layer.weight.data -
                                                            initial_model.input_layer.weight.data)
            Delta_b_1 = (model_.width ** (-model_.a[0])) * (model_.input_layer.bias.data -
                                                            initial_model.input_layer.bias.data)

            init_contrib = (model_.width ** (-model_.a[0])) * initial_model.input_layer.forward(x)
            update_contrib = F.linear(x, Delta_W_1, Delta_b_1)

            base_lrs.append(self.base_lr)

            x = model_.activation(init_contrib + base_lrs[0] * update_contrib)  # should be Theta(1) for large widths

            # intermediate layer grads
            for l in range(2, model_.n_layers):
                layer_key = "layer_{:,}_intermediate".format(l)
                layer = getattr(model_.intermediate_layers, layer_key)
                init_layer = getattr(initial_model.intermediate_layers, layer_key)

                Delta_W = (model_.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)

                init_contrib = initial_model.layer_scales[l - 1] * init_layer.forward(x)
                update_contrib = F.linear(x, Delta_W)

                inv_scale = 1.0 / update_contrib.abs().mean()
                base_lrs.append(inv_scale.item())

                x = model_.activation(init_contrib + inv_scale * update_contrib)  # should be Theta(1)

            # output layer
            Delta_W = (model.width ** (-model_.a[model_.n_layers - 1])) * (model_.output_layer.weight.data -
                                                                           initial_model.output_layer.weight.data)

            init_contrib = initial_model.layer_scales[model_.n_layers - 1] * initial_model.output_layer.forward(x)
            update_contrib = F.linear(x, Delta_W)

            inv_scale = 0.1 / update_contrib.abs().mean()

            base_lrs.append(inv_scale.item())

        logging.info('initial base lrs :', base_lrs)
        return base_lrs

    def _set_param_group_lrs(self, base_lrs: Union[float, list] = None):
        if base_lrs is None:
            base_lrs = [self.base_lr] * len(self.initial_lrs)
        elif isinstance(base_lrs, float):
            base_lrs = [base_lrs] * len(self.initial_lrs)

        for i, param in enumerate(self.optimizer.param_groups):
            param['lr'] = base_lrs[i] * self.current_lrs[i]

    def get_lr(self):
        return self.current_lrs

    def step(self):
        self._last_lrs = self.current_lrs

        # if n_warmup_steps is reached, switch to warm lrs
        if self._step_count == self.n_warmup_steps:  # _step_count starts at 1
            self.current_lrs = self.warm_lrs
            self._set_param_group_lrs(self.base_lr)

        self._step_count += 1


class WarmupSwitchLRBias(WarmupSwitchLR):
    """
    A learning rate scheduler which simply switches from initial learning rate values to warm learning rate values after
    a certain number of warmup steps.
    """
    def __init__(self, optimizer, initial_lrs, warm_lrs, initial_bias_lrs, warm_bias_lrs, n_warmup_steps=1,
                 base_lr=0.01, last_epoch=-1, calibrate_base_lr=True, model=None, batches=None,
                 default_calibration=False):
        self._check_lrs(optimizer, initial_bias_lrs, warm_bias_lrs)

        self.initial_bias_lrs = initial_bias_lrs
        self.warm_bias_lrs = warm_bias_lrs

        self.current_bias_lrs = initial_bias_lrs
        self._last_bias_lrs = None

        super().__init__(optimizer, initial_lrs, warm_lrs, n_warmup_steps, base_lr, last_epoch, calibrate_base_lr,
                         model, batches, default_calibration)

    @staticmethod
    def _check_lrs(optimizer, initial_lrs, warm_lrs):
        # there is one param group for the weights and one for the bias of each layer
        n_param_groups = len(list(optimizer.param_groups)) // 2
        if (len(initial_lrs) != n_param_groups) or (len(warm_lrs) != n_param_groups):
            raise ValueError("There must be as many learning rates as there are parameter groups but there were {:,} "
                             "parameter groups but {:,} initial learning rates and {:,} warm learning rates".
                             format(n_param_groups, len(initial_lrs), len(warm_lrs)))

    def calibrate_base_lr(self, model, batches, normalize_first=True):
        logging.info("Calibrating initial base learning rate")

        # use a copy of the model and optimizer so as not to modify the parameters of the object passed
        model_ = deepcopy(model)
        model_.train()

        base_lr = 1.0
        # set mock SGD optimizer with base_lr = 1.0 so that no additional scaling factor appears
        param_groups = \
            [{'params': model_.input_layer.parameters(), 'lr': base_lr * model_.lr_scales[0]}] + \
            [{'params': layer.parameters(), 'lr': base_lr * model_.lr_scales[l+1]}
             for l, layer in enumerate(model_.intermediate_layers)] + \
            [{'params': model_.output_layer.parameters(), 'lr': base_lr * model_.lr_scales[model_.n_layers - 1]}]

        weight_param_groups = \
            [{'params': model_.input_layer.weight, 'lr': model_.base_lr * model_.lr_scales[0], 'name': 'weights_1'}] + \
            [{'params': layer.weight, 'lr': model_.base_lr * model_.lr_scales[l+1], 'name': 'weights_{}'.format(l+2)}
             for l, layer in enumerate(model_.intermediate_layers)] + \
            [{'params': model_.output_layer.weight, 'lr': model_.base_lr * model_.lr_scales[model_.n_layers - 1],
              'name': 'weights_{}'.format(model_.n_layers)}]

        bias_param_groups = \
            [{'params': model_.input_layer.bias, 'lr': model_.base_lr * model_.bias_lr_scales[0], 'name': 'bias_1'}] + \
            [{'params': layer.bias, 'lr': model_.base_lr * model_.bias_lr_scales[l+1], 'name': 'bias_{}'.format(l+2)}
             for l, layer in enumerate(model_.intermediate_layers)] + \
            [{'params': model_.output_layer.bias, 'lr': model_.base_lr * model_.bias_lr_scales[model_.n_layers - 1],
              'name': 'bias_{}'.format(model_.n_layers)}]

        param_groups = weight_param_groups + bias_param_groups

        optimizer = SGD(param_groups, lr=1.0)

        # remember initial weight values
        initial_model = deepcopy(model_)

        # take first step of optimization
        x, y = batches[0]
        y_hat = model_.forward(x, normalize_first=normalize_first)
        loss = model_.loss(y_hat, y)
        loss.backward()
        optimizer.step()

        # calibrate the lr using activations of second forward pass
        model_.eval()
        x, _ = batches[1]
        base_lrs = []
        with torch.no_grad():
            Delta_W_1 = (model_.width ** (-model_.a[0])) * (model_.input_layer.weight.data -
                                                            initial_model.input_layer.weight.data)
            Delta_b_1 = (model_.width ** (-model_.a[0])) * (model_.input_layer.bias.data -
                                                            initial_model.input_layer.bias.data)

            init_contrib = (model_.width ** (-model_.a[0])) * initial_model.input_layer.forward(x)
            update_contrib = F.linear(x, Delta_W_1, Delta_b_1)

            if normalize_first:
                init_contrib = init_contrib / math.sqrt(model_.d + 1)
                update_contrib = update_contrib / math.sqrt(model_.d + 1)

            # inv_scale = 1.0 / update_contrib.abs().mean()

            # base_lrs.append(inv_scale.item())
            if normalize_first:
                base_lrs.append(self.base_lr * (model_.d + 1))
            else:
                base_lrs.append(self.base_lr)

            x = model_.activation(init_contrib + base_lrs[0] * update_contrib)  # should be Theta(1)

            # intermediate layer grads
            for l in range(2, model_.n_layers):
                layer_key = "layer_{:,}_intermediate".format(l)
                layer = getattr(model_.intermediate_layers, layer_key)
                init_layer = getattr(initial_model.intermediate_layers, layer_key)

                Delta_W = (model.width ** (-model.a[l - 1])) * (layer.weight.data - init_layer.weight.data)
                Delta_b = layer.bias.data - init_layer.bias.data

                init_contrib = init_layer.forward(initial_model.layer_scales[l - 1] * x)
                update_contrib = F.linear(x, Delta_W, Delta_b)

                inv_scale = 1.0 / update_contrib.abs().mean()
                base_lrs.append(inv_scale.item())

                x = model_.activation(init_contrib + inv_scale * update_contrib)  # should be Theta(1)

            # output layer
            Delta_W = (model.width ** (-model_.a[model_.n_layers - 1])) * (model_.output_layer.weight.data -
                                                                           initial_model.output_layer.weight.data)
            Delta_b = model_.output_layer.bias.data - initial_model.output_layer.bias.data

            init_contrib = initial_model.output_layer.forward(initial_model.layer_scales[model_.n_layers - 1] * x)
            update_contrib = F.linear(x, Delta_W, Delta_b)

            # inv_scale = 1.0 / update_contrib.abs().mean()
            inv_scale = 0.1 / update_contrib.abs().mean()
            # inv_scale = 0.01 / update_contrib.abs().mean()

            base_lrs.append(inv_scale.item())

        print('initial base lr :', base_lrs)
        return base_lrs

    def _set_param_group_lrs(self, base_lrs: Union[float, list] = None):
        if base_lrs is None:
            base_lrs = [self.base_lr] * len(self.initial_lrs)
        elif isinstance(base_lrs, float):
            base_lrs = [base_lrs] * len(self.initial_lrs)

        for i, param in enumerate(self.optimizer.param_groups):
            if 'weight' in param['name']:
                param['lr'] = base_lrs[i] * self.current_lrs[i]
            elif 'bias' in param['name']:
                param['lr'] = self.base_lr * self.current_bias_lrs[i - len(self.current_lrs)]
            else:
                raise ValueError("param name dit not contain 'weight' nor 'bias' in param group {:,} of optimizer".
                                 format(i))

    def step(self):
        self._last_lrs = self.current_lrs
        self._last_bias_lrs = self.current_bias_lrs

        # if n_warmup_steps is reached, switch to warm lrs
        if self._step_count == self.n_warmup_steps:  # _step_count starts at 1
            self.current_lrs = self.warm_lrs
            self.current_bias_lrs = self.warm_bias_lrs
            self._set_param_group_lrs(self.base_lr)

        self._step_count += 1


SCHED_DICT = {'step_lr': torch.optim.lr_scheduler.StepLR,
              'exp_lr': torch.optim.lr_scheduler.ExponentialLR,
              'cosine_lr': torch.optim.lr_scheduler.CosineAnnealingLR,
              'warmup_switch': WarmupSwitchLR}
DEFAULT_SCHED = None


def get_scheduler(scheduler=None):
    if scheduler is None:
        return None
    elif isinstance(scheduler, str):
        if scheduler in SCHED_DICT.keys():
            return SCHED_DICT[scheduler]
        else:
            raise ValueError("Scheduler name must be one of {} but was {}".format(list(SCHED_DICT.keys()),
                                                                                  scheduler))
    elif isinstance(scheduler, torch.nn.Module):
        return scheduler
    else:
        raise ValueError("Scheduler argument must be of type None, str, or torch.nn.Module but was of type {}".\
                         format(type(scheduler)))

