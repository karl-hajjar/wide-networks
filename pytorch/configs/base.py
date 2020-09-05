class BaseConfig:
    """
    A class implementing a basic configuration holding all the values of the hyperparameters of a certain object (a
    loss, an optimiser, an activation function, ...).
    """

    def __init__(self, config_dict: dict = None, **kwargs):
        """
        Creates an object of the class BaseConfig either from a config dictionary, or from the values passed in
        **kwargs.
        :param config_dict: a dict containing the name and values of the object parameters. Default None.
        :param kwargs: additional arguments
        """
        if config_dict is not None:
            self._init_from_dict(config_dict)
        else:
            self._init_from_args(**kwargs)

    def _init_from_dict(self, d):
        self.name = d["name"] if "name" in d.keys() else None
        if "params" in d.keys():
            self.params = d["params"]
        else:
            self._create_params_att_from_dict(d)

    def _init_from_args(self, **kwargs):
        self.name = kwargs["name"] if "name" in kwargs.keys() else None
        self._create_params_att_from_dict(kwargs)

    def _create_params_att_from_dict(self, d: dict):
        other_keys = [key for key in d.keys() if key != "name"]
        if len(other_keys) >= 1:
            self.params = {key: d[key] for key in other_keys}
