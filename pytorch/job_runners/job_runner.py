import os


class JobRunner(object):
    """
    A class to run (an) experiment(s) for a given model and architecture.
    """

    LOG_NAME = 'run.log'
    RESULTS_FILE = 'results.pickle'
    SEED = 42

    def __init__(self, config_dict: dict, base_experiment_path: str):
        """
        Defines an instance of the class to be later run.
        :param config_dict: a dictionary holding the configuration of the model (name, architecture, loss, optimizer,
        ect).
        :param base_experiment_path: a string representing the path to where the experiment information will be stored.
        """
        self.config_dict = config_dict
        # define string to represent model
        model_config = 'activation={}_loss={}_opt={}_init={}'.format(config_dict['activation']['name'],
                                                                     config_dict['loss']['name'],
                                                                     config_dict['optimizer']['name'],
                                                                     config_dict['initializer']['name'])
        if ('normalization' in config_dict.keys()) and ('name' in config_dict['normalization'].keys()):
            model_config += '_norm=' + config_dict['normalization']['name']
        self.model_config = model_config

        # define corresponding directory in experiments folder
        self.base_experiment_path = os.path.join(base_experiment_path, model_config)  # base experiment folder

    def run(self):
        pass
