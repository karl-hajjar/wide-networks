#!/bin/bash

n_steps=4000

qsub -v activation="relu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="gelu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="elu",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="tanh",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
# qsub -v activation="sigmoid",dataset="mnist",n_steps=$n_steps scripts/server_scripts/abc_params/training/ipllr.sh
