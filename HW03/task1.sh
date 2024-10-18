#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH -J zhuHW3T1
#SBATCH -o zhuHW3T1.out
#SBATCH -e zhuHW3T1.err

#SBATCH --cpus-per-task=20

output_file="zhuHW3T1.csv"

# Compile the code (assuming mmul.cpp and task1.cpp are in the same directory)
g++ task1.cpp matmul.cpp -O3 -fopenmp -o task1

# Run the program for t = 1 to t = 20
for n in {1..20}; do
  # Run the program and capture the output
  time_taken=$(./task1 1024 $n | head -n 1)

  # Write n and the corresponding time to the output file
  echo "$n,$time_taken" >> $output_file
done