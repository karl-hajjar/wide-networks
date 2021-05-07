#!/bin/bash

qsub -v activation="relu",dataset="cifar10",n_steps=600 scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="gelu",dataset="cifar10",n_steps=600 scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="elu",dataset="cifar10",n_steps=600 scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="tanh",dataset="cifar10",n_steps=600 scripts/server_scripts/abc_params/training/ipllr.sh &
qsub -v activation="sigmoid",dataset="cifar10",n_steps=600 scripts/server_scripts/abc_params/training/ipllr.sh
