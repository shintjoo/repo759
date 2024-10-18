#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH --cpus-per-task=20 
#SBATCH -J zhuHW3T2
#SBATCH -o zhuHW3T2.out
#SBATCH -e zhuHW3T2.err


output_file="zhuHW3T2.csv"

# Compile the code
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Run the program for t = 1 to t = 20
for t in {1..20}; do
  # Run the program and capture the output
  time_taken=$(./task2 1024 $t | head -n 1)

  # Write n and the corresponding time to the output file
  echo "$t,$time_taken" >> $output_file
done