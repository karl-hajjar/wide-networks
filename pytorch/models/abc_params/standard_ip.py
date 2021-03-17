from .base_ip import BaseIP


class StandardIP(BaseIP):
    """
    A base class implementing the general skeleton of the IP parameterization which corresponds to the parameterization
    used in the study of Mean field limits of NNs. Anything that is architecture specific is left out of this class and
    has to be implemented in the child classes.
    """

    def __init__(self, config, width: int = None, results_path=None, base_lr=0.01):
        """
        Base class for the IP parameterization with the standard learning rates of Mean Field models:
         - a[0] = 0, a[l] = 1 for l in [1,L-1]
         - b[l] = 0, for any l in [0, L-1]
         - c[0] = -1, c[l] = -2 for l in [1,L-2], c[L-1] = -1
        :param config: the configuration to define the network (architecture, loss, optimizer)
        :param width: the common width (number of neurons) of all layers except the last.
        """
        L = self.n_layers
        c = [-1] + [-2 for _ in range(1, L-1)] + [-1]

        super().__init__(config, c, width, base_lr, results_path)

    def _get_opt_lr(self):
        # intermediate layers, indexed from 1 to self.n_layers - 2 have the same lr in the standard setting for IPs
        lrs = [self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr'],
               self.optimizer.param_groups[self.n_layers - 1]['lr']]
        return lrs
