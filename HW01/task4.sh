#!/usr/bin/env zsh

#SBATCH -p instruction

# 2 CPU cores
#SBATCH --cpus-per-task=2

# A job name of FirstSlurm
#SBATCH --job-name=FirstSlurm

# An output file called FirstSlurm.out
#SBATCH --output="FirstSlurm.out"

# An error file called FirstSlurm.err
#SBATCH --error=FirstSlurm.err