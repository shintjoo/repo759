#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

# 2 CPU cores
#SBATCH --cpus-per-task=2

# A job name of FirstSlurm
#SBATCH -J FirstSlurm

# An output file called FirstSlurm.out
#SBATCH -o FirstSlurm.out

# An error file called FirstSlurm.err
#SBATCH -e FirstSlurm.err

#Run a single command to print the hostname of the machine (compute node) running the job.
hostname