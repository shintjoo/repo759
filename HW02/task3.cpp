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

int main(int argc, char** argv) {
    // Local variable declaration
    int n = 1024;

    double *A = (double *)malloc(sizeof(double)*n*n);
    double *B = (double *)malloc(sizeof(double)*n*n);
    double *C1 = (double *)malloc(sizeof(double)*n*n);
    double *C2 = (double *)malloc(sizeof(double)*n*n);
    double *C3 = (double *)malloc(sizeof(double)*n*n);
    double *C4 = (double *)malloc(sizeof(double)*n*n);
    std::vector<double> vectorA;
    std::vector<double> vectorB;

    // Variables needed for the Mersenne Twister Enginer
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distA(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distB(-1.0f, 1.0f); // need a seperate generator for the 2nd input array so it is different

    for (int i = 0; i < n*n; i++) {
        A[i] = distA(generator);
        B[i] = distB(generator);
        vectorA.push_back(A[i]);
        vectorB.push_back(B[i]);
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

    // Time mmul4
    auto start_time_mmul4 = std::chrono::high_resolution_clock::now();
    mmul4(vectorA, vectorB, C4, n);
    auto end_time_mmul4 = std::chrono::high_resolution_clock::now();
    auto duration_mmul4 = std::chrono::duration<float, std::milli>(end_time_mmul4 - start_time_mmul4);

    std::cout << n << std::endl;
    std::cout << duration_mmul1.count() << std::endl;
    std::cout << C1[n*n-1] << std::endl;
    std::cout << duration_mmul2.count() << std::endl;
    std::cout << C2[n*n-1] << std::endl;
    std::cout << duration_mmul3.count() << std::endl;
    std::cout << C3[n*n-1] << std::endl;
    std::cout << duration_mmul4.count() << std::endl;
    std::cout << C4[n*n-1] << std::endl;    

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;
    delete[] C4;

    return 0;
}