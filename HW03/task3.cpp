/**
 * Shawn Zhu
 * task3.cpp
 * 
 * Credit:
 * 
 */

#include <iostream>
#include <random>
#include <chrono>
#include "matmul.h"
#include <vector>
#include <omp.h>

int main(int argc, char* argv[]) {
    // Check for correct number of arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts" << std::endl;
        return 1;
    }

    // Read arguments from the command line
    std::size_t n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    std::size_t ts = std::atoi(argv[3]);

    // Set number of threads
    omp_set_num_threads(t);

    // Variables needed for the Mersenne Twister Engine
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distA(-1.0f, 1.0f);
}