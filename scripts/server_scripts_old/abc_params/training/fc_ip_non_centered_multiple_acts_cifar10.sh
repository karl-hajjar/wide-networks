#!/bin/bash

n_steps=1200

qsub -v activation="relu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="gelu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="elu",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="tanh",dataset="cifar10",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh
