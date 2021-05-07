import click

from utils.tools import set_random_seeds, set_up_logger, create_dir
from utils.abc_params.debug_ipllr import *
from utils.plot.abc_parameterizations.debug_ipllr import *

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
FIGURES_DIR = os.path.join(ROOT, 'figures/abc_parameterizations/training/')
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', 'fc_ipllr_mnist.yaml')


N_TRIALS = 5
SEED = 30
L = 6
width = 1024
n_warmup_steps = 1
renorm_first = False
scale_first_lr = False


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
    figures_dir = os.path.join(FIGURES_DIR, dataset)
    create_dir(figures_dir)
    log_path = os.path.join(figures_dir, 'log_ipllr_{}.txt'.format(activation))
    logger = set_up_logger(log_path)

    logger.info('Parameters of the run:')
    logger.info('activation = {}'.format(activation))
    logger.info('n_steps = {:,}'.format(n_steps))
    logger.info('base_lr = {}'.format(base_lr))
    logger.info('batch_size = {:,}'.format(batch_size))
    logger.info('dataset = {}'.format(dataset))
    logger.info('Random SEED : {:,}'.format(SEED))
    logger.info('Number of random trials for each model : {:,}'.format(N_TRIALS))

    try:
        set_random_seeds(SEED)  # set random seed for reproducibility
        config_dict = read_yaml(CONFIG_PATH)

        fig_name_template = 'IPLLRs_1_last_small_{}_{}_L={}_m={}_act={}_lr={}_bs={}.png'

        config_dict['architecture']['width'] = width
        config_dict['architecture']['n_layers'] = L + 1
        config_dict['optimizer']['params']['lr'] = base_lr
        config_dict['activation']['name'] = activation
        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': n_warmup_steps,
                                               'calibrate_base_lr': True,
                                               'default_calibration': False}}

        # Load data & define models
        logger.info('Loading data ...')
        if dataset == 'mnist':
            from utils.dataset.mnist import load_data
        elif dataset == 'cifar10':
            from utils.dataset.mnist import load_data
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

        config_dict['scheduler']['params']['calibrate_base_lr'] = False
        config = ModelConfig(config_dict)

        logger.info('Defining models')
        ipllrs = [FcIPLLR(config) for _ in range(N_TRIALS)]

        config_dict['scheduler']['params']['calibrate_base_lr'] = True
        config = ModelConfig(config_dict)
        ipllrs_calib = [FcIPLLR(config, lr_calibration_batches=batches) for _ in range(N_TRIALS)]
        ipllrs_calib_renorm = [FcIPLLR(config, lr_calibration_batches=batches) for _ in range(N_TRIALS)]
        ipllrs_calib_renorm_scale_lr = [FcIPLLR(config, lr_calibration_batches=batches) for _ in range(N_TRIALS)]

        logger.info('Copying parameters of base ipllr')
        for i in range(N_TRIALS):
            ipllrs_calib[i].copy_initial_params_from_model(ipllrs[i])
            ipllrs_calib_renorm[i].copy_initial_params_from_model(ipllrs[i])
            ipllrs_calib_renorm_scale_lr[i].copy_initial_params_from_model(ipllrs[i])

            ipllrs_calib[i].initialize_params()
            ipllrs_calib_renorm[i].initialize_params()
            ipllrs_calib_renorm_scale_lr[i].initialize_params()

        # Make sure calibration takes into account normalization
        logger.info('Recalibrating lrs with new initialisation')
        for ipllr in ipllrs_calib:
            initial_base_lrs = ipllr.scheduler.calibrate_base_lr(ipllr, batches=batches, normalize_first=False)
            ipllr.scheduler._set_param_group_lrs(initial_base_lrs)

        for ipllr in ipllrs_calib_renorm:
            initial_base_lrs = ipllr.scheduler.calibrate_base_lr(ipllr, batches=batches, normalize_first=True)
            ipllr.scheduler._set_param_group_lrs(initial_base_lrs)

        for ipllr in ipllrs_calib_renorm_scale_lr:
            initial_base_lrs = ipllr.scheduler.calibrate_base_lr(ipllr, batches=batches, normalize_first=True)
            ipllr.scheduler._set_param_group_lrs(initial_base_lrs)

        # scale lr of first layer if needed
        for ipllr in ipllrs_calib_renorm_scale_lr:
            ipllr.scheduler.warm_lrs[0] = ipllr.scheduler.warm_lrs[0] * (ipllr.d + 1)

        # with calibration
        results = dict()
        logger.info('Generating training results ...')
        results['ipllr_calib'] = [collect_training_losses(ipllrs_calib[i], batches, n_steps, normalize_first=False)
                                  for i in range(N_TRIALS)]

        results['ipllr_calib_renorm'] = [collect_training_losses(ipllrs_calib_renorm[i], batches, n_steps,
                                                                 normalize_first=True)
                                         for i in range(N_TRIALS)]

        results['ipllr_calib_renorm_scale_lr'] = [
            collect_training_losses(ipllrs_calib_renorm_scale_lr[i], batches, n_steps, normalize_first=True)
            for i in range(N_TRIALS)]

        mode = 'training'
        losses = dict()
        for key, res in results.items():
            losses[key] = [r[0] for r in res]

        chis = dict()
        for key, res in results.items():
            chis[key] = [r[1] for r in res]

        # Plot losses and derivatives
        logger.info('Plotting figures')
        key = 'loss'
        plt.figure(figsize=(12, 8))
        plot_losses_models(losses, key=key, L=L, width=width, activation=activation, lr=base_lr, batch_size=batch_size,
                           mode=mode, normalize_first=renorm_first, marker=None, name='IPLLR')

        plt.savefig(os.path.join(FIGURES_DIR,
                                 fig_name_template.format(mode, key, L, width, activation, base_lr, batch_size)))

        key = 'chi'
        plt.figure(figsize=(12, 8))
        plot_losses_models(chis, key=key, L=L, width=width, activation=activation, lr=base_lr, batch_size=batch_size,
                           mode=mode, marker=None, name='IPLLR')
        plt.savefig(os.path.join(FIGURES_DIR,
                                 fig_name_template.format(mode, key, L, width, activation, base_lr, batch_size)))

    except Exception as e:
        logger.exception("Exception when running the script : {}".format(e))


if __name__ == '__main__':
    main()
