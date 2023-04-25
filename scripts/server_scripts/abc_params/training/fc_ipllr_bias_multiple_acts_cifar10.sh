#!/bin/bash

n_steps=6000

sbatch scripts/server_scripts/abc_params/training/fc_ipllr_bias.sh "relu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ipllr_bias.sh "gelu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ipllr_bias.sh "elu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ipllr_bias.sh "tanh" "cifar10" $n_steps
