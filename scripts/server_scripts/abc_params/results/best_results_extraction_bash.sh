#!/bin/bash

dataset="mnist"
model="ipllr"

sbatch scripts/server_scripts/abc_params/results/best_results_extraction_slurm.sh $dataset $model

