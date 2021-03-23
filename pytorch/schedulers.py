import torch


class WarmupSwitchLR(torch.optim.lr_scheduler._LRScheduler):
    """

    """
    def __init__(self, optimizer, initial_lrs, warm_lrs, n_warmup_steps=1, base_lr=0.01, last_epoch=-1):
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
        super().__init__(optimizer, last_epoch)
        self._set_param_group_lrs()

        # also possible is to reset _step_count to 0 after the call to the parent __init__

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

    def _set_param_group_lrs(self):
        for i, param in enumerate(self.optimizer.param_groups):
            param['lr'] = self.base_lr * self.current_lrs[i]

    def get_lr(self):
        return self.current_lrs

    def step(self):
        self._last_lrs = self.current_lrs
        # if n_warmup_steps is reached, switch to warm lrs
        if self._step_count == self.n_warmup_steps:  # _step_count starts at 1
            self.current_lrs = self.warm_lrs
            self._set_param_group_lrs()

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

