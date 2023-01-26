#!/bin/bash

n_steps=1200

sbatch --export=activation="relu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch --export=activation="gelu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch --export=activation="elu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch --export=activation="tanh",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
# sbatch --export=activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
