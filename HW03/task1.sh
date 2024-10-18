#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH -J zhuHW3T1
#SBATCH -o zhuHW2T1.cout
#SBATCH -e zhuHW2T1.err

#SBATCH --cpus-per-task=20

# Compile the code (assuming mmul.cpp and task1.cpp are in the same directory)
g++ task1.cpp mmul.cpp -O3 -fopenmp -o task1

# Create a file to store timing results
echo "Threads Time(ms)" > task1_times.txt

# Run the program for t = 1 to t = 20
for t in {1..20}; do
    # Run the task1 program with n = 1024 and different thread counts
    # Use /usr/bin/time to measure the execution time
    /usr/bin/time -f "%e" ./task1 1024 $t 2>> task1_times.txt
done