import yaml
from .base import BaseConfig


class ModelConfig:
    """
    A class implementing the configuration holding all the values of the hyperparameters defining a neural network.
    """

    def __init__(self, config_dict: dict = None, config_file: str = None, **kwargs):
        """
        Creates an object of the class config either from a config yaml file, a dictionary, or the values passed in
        **kwargs. If a dict and a file path are both passed as argument, the values inside the dict are used to create
        the config.
        :param config_dict: a dict containing the name and values of the hyperparameters. Default None.
        :param config_file: the path to a yaml file containing the name and values of the hyperparameters. Default None.
        :param kwargs: additional arguments
        """
        self._dict = None
        if (config_dict is None) and (config_file is None):
            self._init_from_args(**kwargs)
        else:
            if config_dict is None:
                # config_file cannot be None here, because config_dict is, and we are in the else clause
                with open(config_file, 'r') as stream:
                    try:
                        config_dict = yaml.safe_load(stream)
                    except yaml.YAMLError as e:
                        raise Exception("Exception while reading yaml file {} : {}".format(config_file, e))
            self._init_from_dict(config_dict)

    def _init_from_dict(self, d):
        if d is None:
            d = dict()
        self._dict = d
        self.name = d["name"] if "name" in d.keys() else "model"
        self.architecture = d["architecture"] if "architecture" in d.keys() else dict()
        # calling BaseConfig() will return and empty config with a single attribute 'name' set to None
        self.activation = BaseConfig(d["activation"]) if "activation" in d.keys() else BaseConfig()
        self.loss = BaseConfig(d["loss"]) if "loss" in d.keys() else BaseConfig()
        self.optimizer = BaseConfig(d["optimizer"]) if "optimizer" in d.keys() else BaseConfig()
        self.initializer = BaseConfig(d["initializer"]) if "initializer" in d.keys() else BaseConfig()
        self.scheduler = BaseConfig(d["scheduler"]) if "scheduler" in d.keys() else None  # scheduler is not mandatory
        if "normalization" in d.keys():
            self.normalization = BaseConfig(d["normalization"])
            if "params" not in d["normalization"]:
                self.normalization.params = dict()
        else:
            self.normalization = None

    def _init_from_args(self, **kwargs):
        self.name = kwargs["name"] if "name" in kwargs.keys() else "model"
        self.activation = self._create_base_config(pattern="activation", **kwargs)
        self.loss = self._create_base_config(pattern="loss", **kwargs)
        self.optimizer = self._create_base_config(pattern="optimizer", **kwargs)
        self.normalization = self._create_base_config(pattern="normalization", **kwargs)
        self.initializer = self._create_base_config(pattern="initializer", **kwargs)
        self.architecture = {key: value for key, value in kwargs.items() if ((key != "name") and
                                                                             ("activation" not in key) and
                                                                             ("loss" not in key) and
                                                                             ("optimizer" not in key) and
                                                                             ("normalization" not in key) and
                                                                             ("scheduler" not in key))}

    @staticmethod
    def _create_base_config(pattern: str, **kwargs):
        if (kwargs is None) or (len(kwargs) == 0) or (pattern not in kwargs.keys()):
            config_dict = None
        else:
            suffix = '_' + pattern  # e.g. looking for "..._loss" in keys of dict
            config_dict = {'name': kwargs[pattern],
                           'params': {key.split(suffix)[0]: value for key, value in kwargs.items() if suffix in key}}
        # calling BaseConfig(None) will return and empty config with a single attribute 'name' set to None
        return BaseConfig(config_dict)

    def dict(self):
        if self._dict is None:
            self._dict = dict()
            self._dict["name"] = self.name
            if len(self.architecture) > 0:
                self._dict["architecture"] = self.architecture
            self._dict["activation"] = self._create_dict(self.activation)
            self._dict["loss"] = self._create_dict(self.loss)
            self._dict["optimizer"] = self._create_dict(self.optimizer)
            self._dict["initializer"] = self._create_dict(self.initializer)
        return self._dict

    @staticmethod
    def _create_dict(pattern_config):
        d = dict()
        if pattern_config.name is not None:
            d["name"] = pattern_config.name
            if (pattern_config.params is not None) and (len(pattern_config.params) > 0):
                d["params"] = pattern_config.params
        else:
            d["name"] = "default"
        return d
