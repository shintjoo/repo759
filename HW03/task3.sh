#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=20 
#SBATCH -J zhuHW3T3
#SBATCH -o zhuHW3T3.out
#SBATCH -e zhuHW3T3.err


output_file="zhuHW3T3_2.csv"

# Compile the code
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Run the program for t = 1 to t = 20
for t in {1..20}; do
  n=$((10**6))
  ts=$((2**6))
  # Run the program and capture the output
  time_taken=$(./task3 $n $t $ts | head -n 1)

  # Write n and the corresponding time to the output file
  echo "$t,$time_taken" >> $output_file
done
