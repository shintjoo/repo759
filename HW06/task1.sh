#!/usr/bin/env zsh

#SBATCH -J zhuHW6T1
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o zhuHW6T1.out -e zhuHW6T1.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
rm -f task1
rm -f zhuHW6T1_1024.out zhuHW6T1_512.out

# Output files for time data
TIME_1024="zhuHW6T1_1024.out"
TIME_512="zhuHW6T1_512.out"

# Compile the program
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Run task1 for n = 2^5, 2^6, ..., 2^14
for i in {5..14}; do
    n=$((2**i))

    # Run with threads_per_block = 1024
    TIME=$(./task1 $n 1024 | tail -n 1)
    echo "$TIME" >> $TIME_1024

    # Run with threads_per_block = 512
    TIME=$(./task1 $n 512 | tail -n 1)
    echo "$TIME" >> $TIME_512
done
