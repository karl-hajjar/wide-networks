print('Hello from ranks_ipllr.py')
import os

cwd = os.getcwd()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'pytorch/configs/abc_parameterizations', 'fc_ipllr_mnist.yaml')

from utils.tools import read_yaml, set_random_seeds
from utils.abc_params.debug_ipllr import *
import sympy


def main():
    print('ROOT :', ROOT)
    print('CONFIG_PATH :', CONFIG_PATH)

    # constants
    SEED = 30
    L = 6
    width = 1024
    n_warmup_steps = 1
    batch_size = 512
    base_lr = 0.1

    set_random_seeds(SEED)  # set random seed for reproducibility
    config_dict = read_yaml(CONFIG_PATH)

    config_dict['architecture']['width'] = width
    config_dict['architecture']['n_layers'] = L + 1
    config_dict['optimizer']['params']['lr'] = base_lr
    config_dict['scheduler'] = {'name': 'warmup_switch',
                                'params': {'n_warmup_steps': n_warmup_steps,
                                           'calibrate_base_lr': True,
                                           'default_calibration': False}}

    base_model_config = ModelConfig(config_dict)

    # Load data & define model

    training_dataset, test_dataset = load_data(download=False, flatten=True)
    train_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
    batches = list(train_data_loader)

    full_x = torch.cat([a for a, _ in batches], dim=0)
    full_y = torch.cat([b for _, b in batches], dim=0)

    # Define model

    ipllr = FcIPLLR(base_model_config, n_warmup_steps=12, lr_calibration_batches=batches)
    ipllr.scheduler.warm_lrs[0] = ipllr.scheduler.warm_lrs[0] * (ipllr.d + 1)

    # Save initial model : t=0
    ipllr_0 = deepcopy(ipllr)

    # Train model one step : t=1
    x, y = batches[0]
    train_model_one_step(ipllr, x, y, normalize_first=True)
    ipllr_1 = deepcopy(ipllr)

    # Train model for a second step : t=2
    x, y = batches[1]
    train_model_one_step(ipllr, x, y, normalize_first=True)
    ipllr_2 = deepcopy(ipllr)

    ipllr.eval()
    ipllr_0.eval()
    ipllr_1.eval()
    ipllr_2.eval()

    layer_scales = ipllr.layer_scales
    intermediate_layer_keys = ["layer_{:,}_intermediate".format(l) for l in range(2, L + 1)]

    # Define W0 and b0
    with torch.no_grad():
        W0 = {1: layer_scales[0] * ipllr_0.input_layer.weight.data.detach() / math.sqrt(ipllr_0.d + 1)}
        for i, l in enumerate(range(2, L + 1)):
            layer = getattr(ipllr_0.intermediate_layers, intermediate_layer_keys[i])
            W0[l] = layer_scales[l - 1] * layer.weight.data.detach()

        W0[L + 1] = layer_scales[L] * ipllr_0.output_layer.weight.data.detach()

    with torch.no_grad():
        b0 = layer_scales[0] * ipllr_0.input_layer.bias.data.detach() / math.sqrt(ipllr_0.d + 1)

    # Define Delta_W_1 and Delta_b_1
    with torch.no_grad():
        Delta_W_1 = {1: layer_scales[0] * (ipllr_1.input_layer.weight.data.detach() -
                                           ipllr_0.input_layer.weight.data.detach()) / math.sqrt(ipllr_1.d + 1)}
        for i, l in enumerate(range(2, L + 1)):
            layer_1 = getattr(ipllr_1.intermediate_layers, intermediate_layer_keys[i])
            layer_0 = getattr(ipllr_0.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_1[l] = layer_scales[l - 1] * (layer_1.weight.data.detach() -
                                                  layer_0.weight.data.detach())

        Delta_W_1[L + 1] = layer_scales[L] * (ipllr_1.output_layer.weight.data.detach() -
                                              ipllr_0.output_layer.weight.data.detach())

    with torch.no_grad():
        Delta_b_1 = layer_scales[0] * (ipllr_1.input_layer.bias.data.detach() -
                                       ipllr_0.input_layer.bias.data.detach()) / math.sqrt(ipllr_1.d + 1)

    # Define Delta_W_2
    with torch.no_grad():
        Delta_W_2 = {1: layer_scales[0] * (ipllr_2.input_layer.weight.data.detach() -
                                           ipllr_1.input_layer.weight.data.detach()) / math.sqrt(ipllr_2.d + 1)}
        for i, l in enumerate(range(2, L + 1)):
            layer_2 = getattr(ipllr_2.intermediate_layers, intermediate_layer_keys[i])
            layer_1 = getattr(ipllr_1.intermediate_layers, intermediate_layer_keys[i])
            Delta_W_2[l] = layer_scales[l - 1] * (layer_2.weight.data.detach() -
                                                  layer_1.weight.data.detach())

        Delta_W_2[L + 1] = layer_scales[L] * (ipllr_2.output_layer.weight.data.detach() -
                                              ipllr_1.output_layer.weight.data.detach())

    with torch.no_grad():
        Delta_b_2 = layer_scales[0] * (ipllr_2.input_layer.bias.data.detach() -
                                       ipllr_1.input_layer.bias.data.detach()) / math.sqrt(ipllr_1.d + 1)

    # Ranks
    print('computing sympy Matrix ...')
    M = sympy.Matrix(Delta_W_1[1].numpy().tolist())

    print('Computing row echelon form ...')
    row_echelon = M.rref()
    print(row_echelon)
    print(row_echelon[1])
    print(len(row_echelon[1]))


if __name__ == '__main__':
    print('Before main')
    main()
