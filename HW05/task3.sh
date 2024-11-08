#!/usr/bin/env zsh

#SBATCH -J zhuHW5T3
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o zhuHW5T3.out -e zhuHW5T3.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
rm -f task3

# Compile program
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
nvcc task316.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task316

# Run program
for exp in {10..29}; do
    n=$((2**exp))
    time_taken_512=$(./task3 $n | head -n 1)
    time_taken_16=$(./task316 $n | head -n 1)
    echo $time_taken_512 >> zhuHW5T3.out
    echo $time_taken_16 >> zhuHW5T316.out

done