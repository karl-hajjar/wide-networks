#!/bin/bash

n_steps=6000

sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "relu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "gelu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "elu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "tanh" "cifar10" $n_steps
