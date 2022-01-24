#!/bin/bash
#SBATCH --job-name=cuda_ising3
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=%x.out
#SBATCH --error=%x.out

./cuda_ising3 4096 128 32