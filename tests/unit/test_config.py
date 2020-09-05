import unittest
import os

from pytorch.configs.model import ModelConfig

RESOURCES_DIR = '../resources/'


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.empty_config_path = os.path.join(RESOURCES_DIR, 'empty_config.yaml')
        self.named_config_path = os.path.join(RESOURCES_DIR, 'named_config.yaml')
        self.reset_config_path = os.path.join(RESOURCES_DIR, 'resnet_config.yaml')

    def test_empty_config_from_file(self):
        config = ModelConfig(config_file=self.empty_config_path)
        self.assertTrue(config.name == "model")
        # activation
        self.assertTrue(config.activation.name is None)
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name is None)
        self.assertFalse(hasattr(config.loss, "params"))
        # opt
        self.assertTrue(config.optimizer.name is None)
        self.assertFalse(hasattr(config.optimizer, "params"))
        # norm
        self.assertTrue(config.normalization is None)

    def test_empty_config_from_dict(self):
        config = ModelConfig(config_dict=dict())
        self.assertTrue(config.name == "model")
        # activation
        self.assertTrue(config.activation.name is None)
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name is None)
        self.assertFalse(hasattr(config.loss, "params"))
        # opt
        self.assertTrue(config.optimizer.name is None)
        self.assertFalse(hasattr(config.optimizer, "params"))
        # norm
        self.assertTrue(config.normalization is None)

    def test_named_config_from_file(self):
        config = ModelConfig(config_file=self.named_config_path)
        self.assertTrue(config.name == "Config")
        # activation
        self.assertTrue(config.activation.name is None)
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name is None)
        self.assertFalse(hasattr(config.loss, "params"))
        # opt
        self.assertTrue(config.optimizer.name is None)
        self.assertFalse(hasattr(config.optimizer, "params"))
        # norm
        self.assertTrue(config.normalization is None)

    def test_named_config_from_dict(self):
        config = ModelConfig(config_dict={"name": "Config"})
        self.assertTrue(config.name == "Config")
        # activation
        self.assertTrue(config.activation.name is None)
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name is None)
        self.assertFalse(hasattr(config.loss, "params"))
        # opt
        self.assertTrue(config.optimizer.name is None)
        self.assertFalse(hasattr(config.optimizer, "params"))
        # norm
        self.assertTrue(config.normalization is None)

    def test_resnet_config_from_file(self):
        config = ModelConfig(config_file=self.reset_config_path)
        self.assertTrue(config.name == "ResNet")
        # architecture
        self.assertDictEqual({"input_size": 28,
                              "n_blocks": 2,
                              "n_layers": 2,
                              "kernel_size": 3,
                              "stride": 1,
                              "n_channels": 32,
                              "bias_conv": True,
                              "fc_dim": 256,
                              "bias_fc": True,
                              "output_size": 10},
                             config.architecture)
        # activation
        self.assertTrue(config.activation.name == "relu")
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name == "cross_entropy")
        self.assertDictEqual(config.loss.params, {"reduction": "mean"})
        # opt
        self.assertTrue(config.optimizer.name == "adam")
        self.assertDictEqual(config.optimizer.params, {"lr":1.0e-4, "beta1": 0.9, "beta2": 0.999})
        # norm
        self.assertTrue(config.normalization.name == "batch_norm_2d")
        self.assertFalse(hasattr(config.normalization, "params"))

    def test_resnet_config_from_dict(self):
        config_dict = {
            "name": "ResNet",
            "activation": {
                "name": "relu"
            },
            "architecture": {
                "input_size": 28,
                "n_blocks": 2,
                "n_layers": 2,
                "kernel_size": 3,
                "stride": 1,
                "n_channels": 32,
                "bias_conv": True,
                "fc_dim": 256,
                "bias_fc": True,
                "output_size": 10
            },
            "loss": {
                "name": "cross_entropy",
                "params": {
                    "reduction": "mean"
                }
            },
            "optimizer": {
                "name": "adam",
                "params": {
                    "lr": 1.0e-4,
                    "beta1": 0.9,
                    "beta2": 0.999
                }
            },
            "normalization": {
                "name": "batch_norm_2d"
            }
        }
        config = ModelConfig(config_dict=config_dict)
        self.assertTrue(config.name == "ResNet")
        # architecture
        self.assertDictEqual({"input_size": 28,
                              "n_blocks": 2,
                              "n_layers": 2,
                              "kernel_size": 3,
                              "stride": 1,
                              "n_channels": 32,
                              "bias_conv": True,
                              "fc_dim": 256,
                              "bias_fc": True,
                              "output_size": 10},
                             config.architecture)
        # activation
        self.assertTrue(config.activation.name == "relu")
        self.assertFalse(hasattr(config.activation, "params"))
        # loss
        self.assertTrue(config.loss.name == "cross_entropy")
        self.assertDictEqual(config.loss.params, {"reduction": "mean"})
        # opt
        self.assertTrue(config.optimizer.name == "adam")
        self.assertDictEqual(config.optimizer.params, {"lr": 1.0e-4, "beta1": 0.9, "beta2": 0.999})
        # norm
        self.assertTrue(config.normalization.name == "batch_norm_2d")
        self.assertFalse(hasattr(config.normalization, "params"))


if __name__ == '__main__':
    unittest.main()
