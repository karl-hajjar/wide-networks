#!/bin/bash

n_steps=6000

sbatch scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh "relu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh "gelu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh "elu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh "tanh" "cifar10" $n_steps
