#!/bin/bash

# Définition de l'environnement parallèle (4 ici pour 4 coeurs) :
#$ -pe make 16

# Nom du calcul, répertoire de travail :
#$ -N "ranks_ipllr"
#$ -wd /workdir2/hajjar/projects/wide-networks

# Optionnel, être notifié par email :
#$ -m abe
#$ -M hajjarkarl@gmail.com

#$ -e error.txt
#$ -o output.txt
#$ -j y

echo $PWD

echo "Loading anaconda ..."
module load anaconda/2020.07  # load anaconda module

echo "Activating virtualenv ..."
source activate karl-wide  # activate the virtual environment

echo "Exporting python path for library wide-networks ..."
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

echo "Launching python script ..."
python3 scripts/server_scripts/ranks_ipllr.py

