import click
import torch
from torch.utils.data import DataLoader

from utils.tools import read_yaml, set_random_seeds, set_up_logger, create_dir
from pytorch.configs.model import ModelConfig
from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
from utils.plot.abc_parameterizations.debug_ipllr import *
from utils.plot.abc_parameterizations.ranks import *
from utils.abc_params.debug_ipllr import *

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
FIGURES_DIR = os.path.join(ROOT, 'figures/abc_parameterizations/ranks/')
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations')


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
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
              help='Which learning rate to use')
@click.option('--batch_size', '-bs', required=False, type=click.INT, default=512,
              help='What batch size to use')
@click.option('--dataset', '-ds', required=False, type=click.STRING, default="mnist",
              help='Which dataset to train on')
def main(activation="relu", base_lr=0.01, batch_size=512, dataset="mnist"):
    config_path = os.path.join(CONFIG_PATH, 'fc_ipllr_{}.yaml'.format(dataset))
    figures_dir = os.path.join(FIGURES_DIR, dataset)
    create_dir(figures_dir)
    log_path = os.path.join(figures_dir, 'log_ipllr_{}.txt'.format(activation))
    logger = set_up_logger(log_path)

    logger.info('Parameters of the run:')
    logger.info('activation = {}'.format(activation))
    logger.info('base_lr = {}'.format(base_lr))
    logger.info('batch_size = {:,}'.format(batch_size))
    logger.info('Random SEED : {:,}'.format(SEED))
    logger.info('Number of random trials for each model : {:,}'.format(N_TRIALS))

    try:
        set_random_seeds(SEED)  # set random seed for reproducibility
        config_dict = read_yaml(config_path)

        version = 'L={}_m={}_act={}_lr={}_bs={}'.format(L, width, activation, base_lr, batch_size)
        template_name = 'ipllr_{}_ranks_{}_' + version
        
        config_dict['architecture']['width'] = width
        config_dict['architecture']['n_layers'] = L + 1
        config_dict['optimizer']['params']['lr'] = base_lr
        config_dict['activation']['name'] = activation
        config_dict['scheduler'] = {'name': 'warmup_switch',
                                    'params': {'n_warmup_steps': n_warmup_steps,
                                               'calibrate_base_lr': True,
                                               'default_calibration': False}}

        config = ModelConfig(config_dict)

        # Load data & define models
        logger.info('Loading data ...')
        if dataset == 'mnist':
            from utils.dataset.mnist import load_data
        elif dataset == 'cifar10':
            from utils.dataset.cifar10 import load_data
        elif dataset == 'cifar100':
            # TODO : add cifar100 to utils.dataset
            pass
        else:
            error = ValueError("dataset must be one of ['mnist', 'cifar10', 'cifar100'] but was {}".format(dataset))
            logger.error(error)
            raise error

        training_dataset, test_dataset = load_data(download=False, flatten=True)
        train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
        batches = list(train_data_loader)

        full_x = torch.cat([a for a, _ in batches], dim=0)
        full_y = torch.cat([b for _, b in batches], dim=0)

        logger.info('Defining models')
        ipllrs = [FcIPLLR(config, lr_calibration_batches=batches) for _ in range(N_TRIALS)]

        for ipllr in ipllrs:
            ipllr.scheduler.warm_lrs[0] = ipllr.scheduler.warm_lrs[0] * (ipllr.d + 1)

        # save initial models
        ipllrs_0 = [deepcopy(ipllr) for ipllr in ipllrs]

        # train model one step
        logger.info('Training model a first step (t=1)')
        x, y = batches[0]
        ipllrs_1 = []
        for ipllr in ipllrs:
            train_model_one_step(ipllr, x, y, normalize_first=True)
            ipllrs_1.append(deepcopy(ipllr))

        # train models for a second step
        logger.info('Training model a second step (t=2)')
        x, y = batches[1]
        ipllrs_2 = []
        for ipllr in ipllrs:
            train_model_one_step(ipllr, x, y, normalize_first=True)
            ipllrs_2.append(deepcopy(ipllr))

        # set eval mode for all models
        for i in range(N_TRIALS):
            ipllrs[i].eval()
            ipllrs_0[i].eval()
            ipllrs_1[i].eval()
            ipllrs_2[i].eval()

        logger.info('Storing initial and update matrices')
        # define W0 and b0
        W0s = []
        b0s = []
        for ipllr_0 in ipllrs_0:
            W0, b0 = get_W0_dict(ipllr_0, normalize_first=True)
            W0s.append(W0)
            b0s.append(b0)

        # define Delta_W_1 and Delta_b_1
        Delta_W_1s = []
        Delta_b_1s = []
        for i in range(N_TRIALS):
            Delta_W_1, Delta_b_1 = get_Delta_W1_dict(ipllrs_0[i], ipllrs_1[i], normalize_first=True)
            Delta_W_1s.append(Delta_W_1)
            Delta_b_1s.append(Delta_b_1)

        # define Delta_W_2 and Delta_b_2
        Delta_W_2s = []
        Delta_b_2s = []
        for i in range(N_TRIALS):
            Delta_W_2, Delta_b_2 = get_Delta_W1_dict(ipllrs_1[i], ipllrs_2[i], normalize_first=True)
            Delta_W_2s.append(Delta_W_2)
            Delta_b_2s.append(Delta_b_2)

        x, y = full_x, full_y  # compute pre-activations on full batch

        # contributions after first step
        h0s = []
        delta_h_1s = []
        h1s = []
        x1s = []
        for i in range(N_TRIALS):
            h0, delta_h_1, h1, x1 = get_contributions_1(x, ipllrs_1[i], W0s[i], b0s[i], Delta_W_1s[i], Delta_b_1s[i],
                                                        normalize_first=True)
            h0s.append(h0)
            delta_h_1s.append(delta_h_1)
            h1s.append(h0)
            x1s.append(x1)

        # ranks of initial weight matrices and first two updates
        logger.info('Computing ranks of weight matrices ...')
        weight_ranks_dfs_dict = dict()

        tol = None
        weight_ranks_dfs_dict['svd_default'] = [get_svd_ranks_weights(W0s[i], Delta_W_1s[i], Delta_W_2s[i], L, tol=tol)
                                                for i in range(N_TRIALS)]

        tol = 1e-7
        weight_ranks_dfs_dict['svd_tol'] = [get_svd_ranks_weights(W0s[i], Delta_W_1s[i], Delta_W_2s[i], L, tol=tol)
                                            for i in range(N_TRIALS)]

        weight_ranks_dfs_dict['squared_tr'] = [get_square_trace_ranks_weights(W0s[i], Delta_W_1s[i], Delta_W_2s[i], L)
                                               for i in range(N_TRIALS)]

        weight_ranks_df_dict = {key: get_concatenated_ranks_df(weight_ranks_dfs_dict[key])
                                for key in weight_ranks_dfs_dict.keys()}
        avg_ranks_df_dict = {key: get_avg_ranks_dfs(weight_ranks_df_dict[key])
                             for key in weight_ranks_df_dict.keys()}

        logger.info('Saving weights ranks data frames to csv ...')
        for key in weight_ranks_df_dict.keys():
            logger.info(key)
            logger.info('\n' + str(avg_ranks_df_dict[key]) + '\n\n')
            avg_ranks_df_dict[key].to_csv(os.path.join(figures_dir, template_name.format(key, 'weights') + '.csv'),
                                          header=True, index=True)

        ranks_dfs = [weight_ranks_df_dict['svd_default'],
                     weight_ranks_df_dict['svd_tol'],
                     weight_ranks_df_dict['squared_tr']]

        # plot weights ranks
        logger.info('Plotting weights ranks')
        plt.figure(figsize=(12, 6))
        plot_weights_ranks_vs_layer('W0', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('W0', 'weights') + '.png'))

        plt.figure(figsize=(12, 6))
        plot_weights_ranks_vs_layer('Delta_W_1', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('Delta_W_1', 'weights') + '.png'))

        plt.figure(figsize=(12, 6))
        plot_weights_ranks_vs_layer('Delta_W_2', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('Delta_W_2', 'weights') + '.png'))

        # ranks of the pre-activations
        logger.info('Computing ranks of (pre-)activations ...')
        act_ranks_dfs_dict = dict()

        tol = None
        act_ranks_dfs_dict['svd_default'] = [get_svd_ranks_acts(h0s[i], delta_h_1s[i], h1s[i], x1s[i], L, tol=tol)
                                             for i in range(N_TRIALS)]

        tol = 1e-7
        act_ranks_dfs_dict['svd_tol'] = [get_svd_ranks_acts(h0s[i], delta_h_1s[i], h1s[i], x1s[i], L, tol=tol)
                                         for i in range(N_TRIALS)]

        act_ranks_dfs_dict['squared_tr'] = [get_square_trace_ranks_acts(h0s[i], delta_h_1s[i], h1s[i], x1s[i], L)
                                            for i in range(N_TRIALS)]

        act_ranks_df_dict = {key: get_concatenated_ranks_df(act_ranks_dfs_dict[key])
                             for key in act_ranks_dfs_dict.keys()}
        avg_ranks_df_dict = {key: get_avg_ranks_dfs(act_ranks_df_dict[key])
                             for key in act_ranks_df_dict.keys()}

        logger.info('Saving (pre-)activation ranks data frames to csv ...')
        for key in avg_ranks_df_dict.keys():
            logger.info(key)
            logger.info('\n' + str(avg_ranks_df_dict[key]) + '\n\n')
            avg_ranks_df_dict[key].to_csv(os.path.join(figures_dir, template_name.format(key, 'acts') + '.csv'),
                                          header=True, index=True)

        ranks_dfs = [act_ranks_df_dict['svd_default'],
                     act_ranks_df_dict['svd_tol'],
                     act_ranks_df_dict['squared_tr']]

        logger.info('Plotting (pre-)activation ranks')
        plt.figure(figsize=(12, 6))
        plot_acts_ranks_vs_layer('h0', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('h0', 'acts') + '.png'))

        plt.figure(figsize=(12, 6))
        plot_acts_ranks_vs_layer('h1', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('h1', 'acts') + '.png'))

        plt.figure(figsize=(12, 6))
        plot_acts_ranks_vs_layer('x1', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('x1', 'acts') + '.png'))

        plt.figure(figsize=(12, 6))
        plot_acts_ranks_vs_layer('delta_h_1', ranks_dfs, tol, L, width, base_lr, batch_size, y_scale='log')
        plt.savefig(os.path.join(figures_dir, template_name.format('delta_h_1', 'acts') + '.png'))

        # diversity in terms of the index of the maximum entry
        logger.info('Computing diversity of the maximum entry of pre-activations...')
        max_acts_diversity_dfs = [get_max_acts_diversity(h0s[i], delta_h_1s[i], h1s[i], L) for i in range(N_TRIALS)]
        max_acts_diversity_df = get_concatenated_ranks_df(max_acts_diversity_dfs)
        avg_max_acts_diversity_df = get_avg_ranks_dfs(max_acts_diversity_df)
        logger.info('Diversity of the maximum activation index df:')
        logger.info(str(avg_max_acts_diversity_df))
        avg_max_acts_diversity_df.to_csv(os.path.join(figures_dir, 'ipllr_max_acts_' + version + '.csv'),
                                         header=True, index=True)

    except Exception as e:
        logger.exception("Exception when running the script : {}".format(e))


if __name__ == '__main__':
    main()
