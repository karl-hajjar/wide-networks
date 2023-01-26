#!/bin/bash

n_steps=600

qsub -v activation="relu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="gelu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="elu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh &
qsub -v activation="tanh",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/fc_ip_non_centered.sh
