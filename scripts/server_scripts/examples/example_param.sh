#!/bin/bash

n=3

echo "Running sbatch with arguments n=$n and word='Karl'"
sbatch scripts/server_scripts/examples/example.sh $n "Karl"
