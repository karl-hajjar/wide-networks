import torch


SCHED_DICT = {'step_lr': torch.optim.lr_scheduler.StepLR,
              'exp_lr': torch.optim.lr_scheduler.ExponentialLR,
              'cosine_lr': torch.optim.lr_scheduler.CosineAnnealingLR}
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


class WarmupSwitchLR(torch.optim.lr_scheduler._LRScheduler):
    """

    """
    def __init__(self, optimizer, initial_lr, warm_lr, n_warmup_steps=1, base_lr=0.01, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

        self.initial_lr = initial_lr
        self.warm_lr = warm_lr
        self.n_warmup_steps = n_warmup_steps
        self.base_lr = base_lr

        # since we update after each step of optimization, scheduler.step() will be called before opt.step() as it will
        # appear in the training_step() method of the LightningModule
        self._step = -1
        self.current_lr = initial_lr
        self._last_lr = initial_lr

    def get_lr(self):
        return self.current_lr

    def step(self):
        if self._step < self.n_warmup_steps:
            for param in self.optimizer.param_groups:
                par

        self._step += 1
