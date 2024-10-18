#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH -J zhuHW3T1
#SBATCH -o zhuHW3T1.out
#SBATCH -e zhuHW3T1.err


output_file="zhuHW3T1.csv"

# Compile the code
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# Run the program for t = 1 to t = 20
for t in {1..20}; do
  # Run the program and capture the output
  time_taken=$(./task1 1024 $t | head -n 1)

  # Write n and the corresponding time to the output file
  echo "$t,$time_taken" >> $output_file
done