#!/bin/bash

#SBATCH -A jiaoyuling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -J to_hdf5
#SBATCH -o out.log
#SBATCH -e err.log
#SBATCH -p pub

PYTHON_PATH="/home/mawensen/project/miniconda3/bin"
$PYTHON_PATH/python -u to_hdf5.py

