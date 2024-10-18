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
#include "msort.h"
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
    std::uniform_int_distribution<int> distA(-1000, 1000);

    int* arr = new int[n];
    for (std::size_t i = 0; i < n; i++){
        arr[i] = distA(generator);
    }

    // Time msort
    auto start_time = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);

    std::cout << arr[0] << std::endl;
    std::cout << arr[n-1] << std::endl;
    std::cout << duration.count() << " ms" << std::endl;

    std::cout << "Output array:" << std::endl;
    for (size_t i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }

    // Clean up
    delete[] arr;

}