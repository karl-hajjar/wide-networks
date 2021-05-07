#!/bin/bash

qsub -v activation="relu",dataset="mnist",n_steps=600 scripts/server_scripts/abc_params/training/muP.sh &
qsub -v activation="gelu",dataset="mnist",n_steps=600 scripts/server_scripts/abc_params/training/muP.sh &
qsub -v activation="elu",dataset="mnist",n_steps=600 scripts/server_scripts/abc_params/training/muP.sh &
qsub -v activation="tanh",dataset="mnist",n_steps=600 scripts/server_scripts/abc_params/training/muP.sh &
qsub -v activation="sigmoid",dataset="mnist",n_steps=600 scripts/server_scripts/abc_params/training/muP.sh
