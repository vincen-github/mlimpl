#!/bin/bash

#SBATCH -A jiaoyuling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -J ssl
#SBATCH -o out.log
#SBATCH -e err.log
#SBATCH -p gpu


PYTHON_PATH=/home/mawensen/project/miniconda3/bin
$PYTHON_PATH/python -u -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --bs 512 --emb 64 --method simclr
