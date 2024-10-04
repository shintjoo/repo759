#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH --cpus-per-task=1

#SBATCH -J zhuHW2T3

#SBATCH -o zhuHW2T3.out

#SBATCH -e zhuHW2T3.err

#compile task3.cpp and matmul.cpp
g++ matmul.cpp task3.cpp -Wall -O3 -std=c++17 -o task3

#run task
./task3

