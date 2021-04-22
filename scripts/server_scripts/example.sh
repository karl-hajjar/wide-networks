#!/bin/bash
# Définition de l'environnement parallèle (4 ici pour 4 coeurs) :
#$ -pe make 4

pwd

# Nom du calcul, répertoire de travail :
#$ -N "Calcul essai toy script"
#$ -wd /workdir2/hajjar/projects/wide-networks

pwd

# Optionnel, être notifié par email :
#$ -m abe
#$ -M hajjarkarl@gmail.com

# rm -f .output
#$ -e error.txt
#$ -o output.txt
#$ -j y

# n, x and comment arguments are given with qsub cmd:
# qsub -v n=5,x=2.0,comment="Test script" script.sh

ls

source env/bin/activate  # activate the virtual environment

export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

python3 scripts/server_scripts/toy_python_script.py --n=5 --word='Karl'
