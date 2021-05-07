#!/bin/bash

qsub -v activation="relu" scripts/server_scripts/abc_params/training/ipllr_mnist.sh &
qsub -v activation="elu" scripts/server_scripts/abc_params/training/ipllr_mnist.sh &
qsub -v activation="tanh" scripts/server_scripts/abc_params/training/ipllr_mnist.sh &
qsub -v activation="sigmoid" scripts/server_scripts/abc_params/training/ipllr_mnist.sh &
