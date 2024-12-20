#!/usr/bin/env zsh

#SBATCH -J zhuHW5T2
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o zhuHW5T2.out -e zhuHW5T2.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
rm -f task2

# Compile program
nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run program
 ./task2