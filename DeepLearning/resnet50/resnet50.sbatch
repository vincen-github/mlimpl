#!/bin/bash

#SBATCH -A jiaoyuling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -J resnet50_pretraining
#SBATCH -o out.log
#SBATCH -e err.log
#SBATCH -p gpu


PYTHON_PATH=/home/mawensen/project/miniconda3/envs/torch/bin
$PYTHON_PATH/python -u train.py
