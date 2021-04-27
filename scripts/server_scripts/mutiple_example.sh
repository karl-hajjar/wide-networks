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

#$-t 1-4:1

# n, x and comment arguments are given with qsub cmd:
# qsub -v n=5,x=2.0,comment="Test script" script.sh

module load anaconda/2020.07  # load anaconda module

source activate karl-wide  # activate the virtual environment

export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

# how to use arguments with qsub
# python3 scripts/server_scripts/toy_python_script.py --n=$n --word=$word
# qsub -v n=4,word="Ja" scripts/server_scripts/example.sh  ## (no space between the arguments separated by comma)

python3 scripts/server_scripts/toy_python_script.py --n=5 --word="Karl" > out_1.txt
python3 scripts/server_scripts/toy_python_script.py --n=4 --word="Arlk" > out_2.txt
python3 scripts/server_scripts/toy_python_script.py --n=3 --word="Rlka" > out_3.txt
python3 scripts/server_scripts/toy_python_script.py --n=6 --word="Lkar" > out_4.txt

