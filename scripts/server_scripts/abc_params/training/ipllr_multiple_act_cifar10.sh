#!/bin/bash

n_steps=1200

sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps

# sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps &
# sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps &
# sbatch scripts/server_scripts/abc_params/training/ipllr.sh "relu" "cifar10" $n_steps &
# sbatch scripts/server_scripts/abc_params/training/ipllr.sh
# sbatch --export=activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
