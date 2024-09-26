#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:30:00

#SBATCH --cpus-per-task=1

#SBATCH -J zhuHW2T1

#SBATCH -o zhuHW2T1.out

#SBATCH -e zhuHW2T1.err

output_file="zhuHW2T1.out"

#compile task1.cpp and scan.cpp
g++ task1.cpp scan.cpp -Wall -O3 -std=c++17 -o task1

#run task1
for ((i=10; i<=30; i++))
do
  n=$((2**i))
  
  # Run the program and capture the output
  time_taken=$(./task1 $n | head -n 1)  # The first line is the time taken in milliseconds

  # Write n and the corresponding time to the output file
  echo "$n,$time_taken" >> $output_file
done

