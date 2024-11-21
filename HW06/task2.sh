#!/usr/bin/env zsh

#SBATCH -J zhuHW6T2
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o zhuHW6T2.out -e zhuHW6T2.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
rm -f task2
rm -f zhuHW6T2_1024.out zhuHW6T2_512.out

# Output files for time data
TIME_1024="zhuHW6T2_1024.out"
TIME_512="zhuHW6T2_512.out"

# Compile the program
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

# Run task1 for n = 2^10, 2^1, ..., 2^29
for i in {10..29}; do
    n=$((2**i))

    # Run with threads_per_block = 1024
    TIME=$(./task2 $n 128 1024 | tail -n 1)
    echo "$TIME" >> $TIME_1024

    # Run with threads_per_block = 512
    TIME=$(./task2 $n 128 512 | tail -n 1)
    echo "$TIME" >> $TIME_512
done
