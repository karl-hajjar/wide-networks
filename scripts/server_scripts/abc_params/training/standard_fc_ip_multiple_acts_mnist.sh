#!/bin/bash

n_steps=6000

sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "relu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "gelu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "elu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "tanh" "cifar10" $n_steps
