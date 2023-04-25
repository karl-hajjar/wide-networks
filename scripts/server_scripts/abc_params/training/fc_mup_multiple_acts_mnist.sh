#!/bin/bash

n_steps=5000

sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "relu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "gelu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "elu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/fc_mup.sh "tanh" "mnist" $n_steps
