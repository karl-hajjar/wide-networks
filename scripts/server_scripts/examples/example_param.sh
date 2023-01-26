#!/bin/bash

echo "Running sbatch with arguments word='Karl'"
# sbatch --export=word="Karl" scripts/server_scripts/examples/example.sh
# sbatch scripts/server_scripts/examples/example.sh
sbatch scripts/server_scripts/examples/example.sh -word="Karl"
