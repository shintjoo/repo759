/**
 * Shawn Zhu
 * task3.cpp
 * 
 * Credit: Completed on my own
 * 
 */

#include <iostream>
#include <random>
#include <chrono>
#include "matmul.h"
#include <vector>

int main(int argc, char** argv) {
    // Local variable declaration
    int n = 1024;

    float *A = new float[n*n];
    float *B = new float[n*n];
    float *C = new float[n*n];

    // Variables needed for the Mersenne Twister Enginer
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distA(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distB(-1.0f, 1.0f); // need a seperate generator for the 2nd input array so it is different

    for (int i = 0; i < n*n; i++) {
        A[i] = distA(generator);
        B[i] = distB(generator);
        C[i] = 0.0;
    }

    // Time mmul
    auto start_time_mmul2 = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    auto end_time_mmul2 = std::chrono::high_resolution_clock::now();
    auto duration_mmul2 = std::chrono::duration<float, std::milli>(end_time_mmul2 - start_time_mmul2);

    std::cout << n << std::endl;
    std::cout << duration_mmul2.count() << std::endl;
    std::cout << C[n*n-1] << std::endl; 

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;


    return 0;
}