#!/usr/bin/env zsh

#SBATCH -J zhuHW7T2
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:30:00
#SBATCH -o zhuHW7T2.out -e zhuHW7T2.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
rm -f zhuHW7T2.out zhuHW7T2.err


# Compile the program
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run task1 for n = 2^5, 2^6, ..., 2^14
for i in {10..29}; do
    n=$((2**i))

    ./task2 $n 256 >> zhuHW7T2_256.out

    ./task2 $n 1024 >> zhuHW7T2_1024.out
done

# Clean up
rm task2