#!/bin/bash
#SBATCH --job-name=seq_ising
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=10:00
#SBATCH --output=%x.out
#SBATCH --error=%x.out

./seq_ising 4096 32

