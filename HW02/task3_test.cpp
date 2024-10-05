/**
 * Shawn Zhu
 * task3_test.cpp
 * 
 * Credit: Completed on my own. Modified test3.cpp
 * 
 */

#include <iostream>
#include <random>
#include <chrono>
#include "matmul.h"
#include <vector>

int main(int argc, char** argv) {
    // Local variable declaration
    const int n = 4;

    double A[n*n] = {45.0, 23.0, 22.0, 15.0, 53.0, 4.0, 3.0, 77.0, 37.0, 6.0, 26.0, 1.0, 37.0, 8.0, 7.0, 2.0};
    double B[n*n] = {10.0, 24.0, 3.0, 75.0, 3.0, 5.0, 34.0, 20.0, 78.0, 5.0, 20.0, 14.0, 55.0, 36.0, 64.0, 21.0};
    double *C1 = new double[n*n];
    double *C2 = new double[n*n];
    double *C3 = new double[n*n];
    double *C4 = new double[n*n];


    // Variables needed for the Mersenne Twister Enginer
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distA(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distB(-1.0f, 1.0f); // need a seperate generator for the 2nd input array so it is different

    for (int i = 0; i < n*n; i++) {
        C1[i] = 0.0;
        C2[i] = 0.0;
        C3[i] = 0.0;
        C4[i] = 0.0;
    }

    // Time mmul1
    auto start_time_mmul1 = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C1, n);
    auto end_time_mmul1 = std::chrono::high_resolution_clock::now();
    auto duration_mmul1 = std::chrono::duration<float, std::milli>(end_time_mmul1 - start_time_mmul1);

    // Time mmul2
    auto start_time_mmul2 = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, n);
    auto end_time_mmul2 = std::chrono::high_resolution_clock::now();
    auto duration_mmul2 = std::chrono::duration<float, std::milli>(end_time_mmul2 - start_time_mmul2);

    // Time mmul3
    auto start_time_mmul3 = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, n);
    auto end_time_mmul3 = std::chrono::high_resolution_clock::now();
    auto duration_mmul3 = std::chrono::duration<float, std::milli>(end_time_mmul3 - start_time_mmul3);


    std::cout << n << std::endl;
    std::cout << duration_mmul1.count() << std::endl;
    std::cout << C1[n*n-1] << std::endl;
    std::cout << duration_mmul2.count() << std::endl;
    std::cout << C2[n*n-1] << std::endl;
    std::cout << duration_mmul3.count() << std::endl;
    std::cout << C3[n*n-1] << std::endl;
    
    std::cout << "Output array 1:" << std::endl;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << C1[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Output array 2:" << std::endl;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << C2[i * n + j] << " ";
        }
        std::cout << std::endl;
    } 
    std::cout << "Output array 3:" << std::endl;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << C3[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    //free(A);
    //free(B);
    delete[] C1;
    delete[] C2;
    delete[] C3;


    return 0;
}