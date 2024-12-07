#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include "matmul.cuh"

// Function to run and time matmul_1 (int)
void runMatMul1(unsigned int n, unsigned int block_dim) {
    // Initialize random number generator for int
    std::mt19937 generator(759);
    std::uniform_int_distribution<int> dist(-100, 100);

    // Allocate and fill matrices
    int *A = new int[n * n];
    int *B = new int[n * n];
    int *C = new int[n * n];

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Testing matmul_1 (int)\n";
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << " ms\n";

    delete[] A;
    delete[] B;
    delete[] C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to run and time matmul_2 (float)
void runMatMul2(unsigned int n, unsigned int block_dim) {
    // Initialize random number generator for float
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate and fill matrices
    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_2(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Testing matmul_2 (float)\n";
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << " ms\n";

    delete[] A;
    delete[] B;
    delete[] C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to run and time matmul_3 (double)
void runMatMul3(unsigned int n, unsigned int block_dim) {
    // Initialize random number generator for double
    std::mt19937 generator(759);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Allocate and fill matrices
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_3(A, B, C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Testing matmul_3 (double)\n";
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << " ms\n";

    delete[] A;
    delete[] B;
    delete[] C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);

    runMatMul1(n, block_dim);
    runMatMul2(n, block_dim);
    runMatMul3(n, block_dim);

    return 0;
}
