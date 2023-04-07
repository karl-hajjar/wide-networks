#!/bin/bash

n_steps=5000

# sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps

sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "gelu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "elu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/ipllr.sh "tanh" "mnist" $n_steps
# sbatch --export=activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh

