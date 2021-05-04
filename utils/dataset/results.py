import os
import pickle

SAVE_DIR = '../../experiments'
NAME = 'wide_2_layer'
RESULTS_FILE = 'results.pickle'

BATCH_SIZES = [128, 64, 32, 1]
N_TRAINs = [5000, 10000, 1000, 512, 256, 128, 64]
Ds = [512, 1000, 5000, 10000, 256, 128, 50, 25, 20, 15, 10, 5]
Ms = [10, 50, 100, 200, 500, 1000, 2000, 5000]


def read_experiments_results(base_dir, batch_sizes=None, n_trains=None, ds=None, ms=None, all_results=None, n_trials=5):
    if all_results is None:
        all_results = dict()

    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    if n_trains is None:
        n_trains = N_TRAINs
    if ds is None:
        ds = Ds
    if ms is None:
        ms = Ms

    for batch_size in batch_sizes:
        for n_train in n_trains:
            for d in ds:
                for m in ms:
                    exp_name = 'bsize={}_ntrain={}_d={}_m={}'.format(batch_size, n_train, d, m)
                    experiment_dir = os.path.join(base_dir, exp_name)
                    if exp_name in all_results.keys():
                        for i in range(1, n_trials + 1):
                            if i > len(all_results[exp_name]):
                                results_path = os.path.join(experiment_dir, 'trial_' + str(i), RESULTS_FILE)
                                if os.path.exists(results_path):
                                    with open(results_path, 'rb') as file:
                                        all_results[exp_name].append(pickle.load(file))
                    else:
                        if os.path.exists(experiment_dir):
                            all_results[exp_name] = []
                        for i in range(1, n_trials + 1):
                            results_path = os.path.join(experiment_dir, 'trial_' + str(i), RESULTS_FILE)
                            if os.path.exists(results_path):
                                with open(results_path, 'rb') as file:
                                    all_results[exp_name].append(pickle.load(file))
    return all_results
