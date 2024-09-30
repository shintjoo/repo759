#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH --cpus-per-task=1

#SBATCH -J zhuHW2T2

#SBATCH -o zhuHW2T2.out

#SBATCH -e zhuHW2T2.err

#compile task2.cpp and convolution.cpp
g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

#run task
./task2 7 5

