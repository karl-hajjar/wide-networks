#!/bin/bash

n_steps=6000

# sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps

sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "gelu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "elu" "cifar10" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "tanh" "cifar10" $n_steps
# sbatch --export=activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
