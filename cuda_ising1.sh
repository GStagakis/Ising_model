#!/bin/bash
#SBATCH --job-name=cuda_ising1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=%x.out
#SBATCH --error=%x.out

./cuda_ising1 4096 32
