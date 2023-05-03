#!/bin/bash

n_steps=5000

sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "relu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "gelu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "elu" "mnist" $n_steps &
sbatch scripts/server_scripts/abc_params/training/standard_fc_ip.sh "tanh" "mnist" $n_steps
