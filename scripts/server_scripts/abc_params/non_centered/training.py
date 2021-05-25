import click
from torch.utils.data import DataLoader

from utils.tools import read_yaml, set_random_seeds, set_up_logger, create_dir
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.standard_fc_ip import StandardFCIP
from pytorch.schedulers import WarmupSwitchLR
from utils.tools import set_random_seeds, set_up_logger, create_dir
from utils.abc_params.debug_ipllr import *
from utils.plot.abc_parameterizations.debug_ipllr import *

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
FIGURES_DIR = os.path.join(ROOT, 'figures/abc_parameterizations/bias/')
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations')


N_TRIALS = 5
SEED = 30
L = 6
width = 1024
n_warmup_steps = 1
renorm_first = True
scale_first_lr = True


@click.command()
@click.option('--activation', '-act', required=False, type=click.STRING, default="relu",
              help='Which activation function to use for the network')
@click.option('--n_steps', '-N', required=False, type=click.INT, default=300,
              help='How many steps of SGD to take')
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
              help='Which learning rate to use')
@click.option('--batch_size', '-bs', required=False, type=click.INT, default=512,
              help='What batch size to use')
@click.option('--dataset', '-ds', required=False, type=click.STRING, default="mnist",
              help='Which dataset to train on')
def main(activation="relu", n_steps=300, base_lr=0.01, batch_size=512, dataset="mnist"):
    config_path = os.path.join(CONFIG_PATH, 'ip_non_centered_{}.yaml'.format(dataset))
    figures_dir = os.path.join(FIGURES_DIR, dataset)
    create_dir(figures_dir)
    log_path = os.path.join(figures_dir, 'log_ip_non_centered_{}.txt'.format(activation))
    logger = set_up_logger(log_path)

    logger.info('Parameters of the run:')
    logger.info('activation = {}'.format(activation))
    logger.info('n_steps = {:,}'.format(n_steps))
    logger.info('base_lr = {}'.format(base_lr))
    logger.info('batch_size = {:,}'.format(batch_size))
    logger.info('Random SEED : {:,}'.format(SEED))
    logger.info('Number of random trials for each model : {:,}'.format(N_TRIALS))

    try:
        set_random_seeds(SEED)  # set random seed for reproducibility
        config_dict = read_yaml(config_path)

        fig_name_template = 'ip_non_centered_{}_{}_L={}_m={}_act={}_lr={}_bs={}.png'

        config_dict['architecture']['width'] = width
        config_dict['architecture']['n_layers'] = L + 1
        config_dict['optimizer']['params']['lr'] = base_lr
        config_dict['activation']['name'] = activation

        base_model_config = ModelConfig(config_dict)

        # Load data & define models
        logger.info('Loading data ...')
        if dataset == 'mnist':
            from utils.dataset.mnist import load_data
        elif dataset == 'cifar10':
            from utils.dataset.cifar10 import load_data
        elif dataset == 'cifar100':
            # TODO : add cifar100 to utils.dataset
            config_dict['architecture']['output_size'] = 100
            pass
        else:
            error = ValueError("dataset must be one of ['mnist', 'cifar10', 'cifar100'] but was {}".format(dataset))
            logger.error(error)
            raise error

        training_dataset, test_dataset = load_data(download=False, flatten=True)
        train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
        batches = list(train_data_loader)

        logger.info('Defining models')
        base_model_config.scheduler = None
        ips_non_centered = [StandardFCIP(base_model_config) for _ in range(N_TRIALS)]
        ips_non_centered_renorm_scale_lr = [StandardFCIP(base_model_config) for _ in range(N_TRIALS)]

        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': n_warmup_steps,
                                               'calibrate_base_lr': True,
                                               'default_calibration': False}}
        config = ModelConfig(config_dict)
        scheduler = config.scheduler
        config.scheduler = None
        ips_non_centered_renorm_scale_lr_calib = [StandardFCIP(config) for _ in range(N_TRIALS)]

        for ip in ips_non_centered_renorm_scale_lr:
            for i, param_group in enumerate(ip.optimizer.param_groups):
                if i == 0:
                    param_group['lr'] = param_group['lr'] * (ip.d + 1)

        for ip in ips_non_centered_renorm_scale_lr_calib:
            for i, param_group in enumerate(ip.optimizer.param_groups):
                if i == 0:
                    param_group['lr'] = param_group['lr'] * (ip.d + 1)

        logger.info('Copying parameters of base ip_non_centered')
        for i in range(N_TRIALS):
            ips_non_centered_renorm_scale_lr[i].copy_initial_params_from_model(ips_non_centered[i])
            ips_non_centered_renorm_scale_lr_calib[i].copy_initial_params_from_model(ips_non_centered[i])

            ips_non_centered_renorm_scale_lr[i].initialize_params()
            ips_non_centered_renorm_scale_lr_calib[i].initialize_params()

        for model in ips_non_centered_renorm_scale_lr_calib:
            model.scheduler = WarmupSwitchLR(model.optimizer, initial_lrs=model.lr_scales, warm_lrs=model.lr_scales,
                                             base_lr=model.base_lr, model=model, batches=batches,
                                             **scheduler.params)

        results = dict()
        logger.info('Generating training results ...')
        results['ip_non_centered'] = [collect_training_losses(ips_non_centered[i], batches, n_steps,
                                                              normalize_first=False)
                          for i in range(N_TRIALS)]

        results['ips_non_centered_renorm_scale_lr'] = [collect_training_losses(ips_non_centered_renorm_scale_lr[i],
                                                                               batches, n_steps, normalize_first=True)
                                 for i in range(N_TRIALS)]

        results['ips_non_centered_renorm_scale_lr_calib'] = \
            [collect_training_losses(ips_non_centered_renorm_scale_lr_calib[i], batches, n_steps, normalize_first=True)
             for i in range(N_TRIALS)]

        mode = 'training'
        losses = dict()
        for key, res in results.items():
            losses[key] = [r[0] for r in res]

        chis = dict()
        for key, res in results.items():
            chis[key] = [r[1] for r in res]

        # Plot losses and derivatives
        logger.info('Saving figures at {}'.format(figures_dir))
        key = 'loss'
        plt.figure(figsize=(12, 8))
        plot_losses_models(losses, key=key, L=L, width=width, activation=activation, lr=base_lr, batch_size=batch_size,
                           mode=mode, normalize_first=renorm_first, marker=None, name='IP-non-centered')
        plt.ylim(0, 2.5)
        plt.savefig(os.path.join(figures_dir,
                                 fig_name_template.format(mode, key, L, width, activation, base_lr, batch_size)))

        key = 'chi'
        plt.figure(figsize=(12, 8))
        plot_losses_models(chis, key=key, L=L, width=width, activation=activation, lr=base_lr, batch_size=batch_size,
                           mode=mode, marker=None, name='IP-non-centered')
        plt.savefig(os.path.join(figures_dir,
                                 fig_name_template.format(mode, key, L, width, activation, base_lr, batch_size)))

    except Exception as e:
        logger.exception("Exception when running the script : {}".format(e))


if __name__ == '__main__':
    main()
