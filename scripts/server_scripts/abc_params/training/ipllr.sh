#!/bin/bash

# -- Nom du calcul, répertoire de travail :
#SBATCH --job-name=karl-ipllr-multiple-cifar
#SBATCH --chdir=/workdir2/hajjar/projects/wide-networks/

# -- Optionnel, pour être notifié par email :
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hajjarkarl@gmail.com

# -- Sortie standard et d'erreur dans le fichier .output :
#SBATCH --output=./%j.stdout
#SBATCH --error=./%j.stderr

# -- Contexte matériel
#SBATCH --nodes=16

echo "Activating virtualenv ..."
source activate karl-wide  # activate the virtual environment

echo "Exporting python path for library wide-networks ..."
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

# -N permet de spécifier le nombre minimum de nœuds (i.e. machine) sur lesquels lancer la commande.
# --nodelist=... permet de spécifier les nœuds à utiliser.
# --gpus=n permet de définir le nombre de gpus à utiliser. --mem-per-gpu=n pour donner la mémoire minimum pour chaque gpu.
# --mem=MB permet de définir le taille minimale de mémoire pour chaque processus.

echo "Launching Python script from bash with srun..."
srun -N16 python3 pytorch/jobs/abc_parameterizations/fc_ipllr_run.py --activation=$activation --dataset=$dataset --n_steps=$n_steps
