#!/bin/bash

# Number of cores to use
#$ -pe make 16

# Nom du calcul, répertoire de travail :
#$ -N "ipllr_training"
#$ -wd /workdir2/hajjar/projects/wide-networks

# Optionnel, être notifié par email :
#$ -m abe
#$ -M hajjarkarl@gmail.com

#$ -e error.txt
#$ -j y

echo "Loading anaconda ..."
module load anaconda/2020.07  # load anaconda module

echo "Activating virtualenv ..."
source activate hajjar-wide  # activate the virtual environment

echo "Exporting python path for library wide-networks ..."
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

echo "Launching python script ..."
python3 pytorch/jobs/abc_parameterizations/fc_ipllr_run.py --activation=$activation --dataset=$dataset --n_steps=$n_steps

