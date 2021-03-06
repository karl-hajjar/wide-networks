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

        self._set_model_version(config_dict)
        self._set_model_config(config_dict)

        # define corresponding directory in experiments folder for the base experiment folder
        self.base_experiment_path = os.path.join(base_experiment_path, self.model_version, self.model_config)

    def _set_model_version(self, config_dict):
        self.model_version = ""

    def _set_model_config(self, config_dict):
        # define string to represent model
        model_config = 'activation={}_loss={}_opt={}_init={}'.format(config_dict['activation']['name'],
                                                                     config_dict['loss']['name'],
                                                                     config_dict['optimizer']['name'],
                                                                     config_dict['initializer']['name'])
        if ('normalization' in config_dict.keys()) and ('name' in config_dict['normalization'].keys()):
            model_config += '_norm=' + config_dict['normalization']['name']
        self.model_config = model_config

    def run(self):
        pass
