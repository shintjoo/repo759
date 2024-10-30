#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=20 
#SBATCH -J zhuHW4T4
#SBATCH -o zhuHW4T4.out
#SBATCH -e zhuHW4T4.err

rm -f task4static task4dynamic task4guided

g++ task4static.cpp -Wall -O3 -std=c++17 -fopenmp -o task4static
g++ task4dynamic.cpp -Wall -O3 -std=c++17 -fopenmp -o task4dynamic
g++ task4guided.cpp -Wall -O3 -std=c++17 -fopenmp -o task4guided

# Parameters for the simulation
N=200    # number of particles
T=100    # simulation end time

# Clear previous results
rm -f zhuHW4T4static.txt zhuHW4T4dynamic.txt zhuHW4T4guided.txt

# Run experiments for each scheduling policy
for t in {1..8}; do
    ./task4static $N $T $t >> zhuHW4T4static.txt
    ./task4dynamic $N $T $t >> zhuHW4T4dynamic.txt
    ./task4guided $N $T $t >> zhuHW4T4guided.txt
done