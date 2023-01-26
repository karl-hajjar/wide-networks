#!/bin/bash

n_steps=1200

srun activation="relu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
srun activation="gelu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
srun activation="elu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
srun activation="tanh",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
# srun activation="sigmoid",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
