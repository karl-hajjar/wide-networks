#!/bin/bash

# Définition de l'environnement parallèle (4 ici pour 4 coeurs) :
#$ -pe make 4

# Nom du calcul, répertoire de travail :
#$ -N "calcul_essai_toy_script"
#$ -wd /workdir2/hajjar/projects/wide-networks

# Optionnel, être notifié par email :
#$ -m abe
#$ -M hajjarkarl@gmail.com

#$ -e error.txt
#$ -o output.txt
#$ -j y

# n, x and comment arguments are given with qsub cmd:
# qsub -v n=5,x=2.0,comment="Test script" script.sh

module load anaconda/2020.07  # load anaconda module

conda activate karl-wide  # activate the virtual environment

export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

python3 scripts/server_scripts/toy_python_script.py --n=5 --word='Karl'
