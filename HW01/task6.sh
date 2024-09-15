#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH --cpus-per-task=1

#SBATCH -J zhuHW1T6

#SBATCH -o zhuHW1T6.out

#SBATCH -e zhuHW1T6.err

#compile task6.cpp
g++ task6.cpp -Wall -O3 -std=c++17 -o task6

#run task6
./task6 6

squeue -u skzhu
echo "Job starts in directory: $(pwd)"
