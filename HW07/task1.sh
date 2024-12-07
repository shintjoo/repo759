#!/usr/bin/env zsh

#SBATCH -J zhuHW7T1
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o zhuHW7T1_14.out -e zhuHW7T1_14.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run task1 for n = 2^5, 2^6, ..., 2^14
for i in {5..14}; do
    n=$((2**i))

    ./task1 $n 8 >> zhuHW7T1.out
done

# Clean up
rm task1