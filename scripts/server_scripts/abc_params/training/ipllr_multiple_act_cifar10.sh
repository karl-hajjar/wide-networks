#!/bin/bash

n_steps=1200

sbatch activation="relu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch activation="gelu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch activation="elu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
sbatch activation="tanh",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
# sbatch activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
