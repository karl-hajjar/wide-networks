#!/bin/bash

qsub -v activation="relu",dataset="mnist" scripts/server_scripts/abc_params/ranks/ipllr.sh &
# qsub -v activation="gelu",dataset="mnist" scripts/server_scripts/abc_params/ranks/ipllr.sh &
qsub -v activation="elu",dataset="mnist" scripts/server_scripts/abc_params/ranks/ipllr.sh &
# qsub -v activation="tanh",dataset="mnist" scripts/server_scripts/abc_params/ranks/ipllr.sh &
# qsub -v activation="sigmoid",dataset="mnist" scripts/server_scripts/abc_params/ranks/ipllr.sh
