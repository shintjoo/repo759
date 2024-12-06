#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <ctime>
#include "matmul.cuh"

// Initialize separate Mersenne Twister generators
std::mt19937 generator(759);
std::uniform_real_distribution<float> dist_float(-1.0f, 1.0f);
std::uniform_int_distribution<int> dist_int(-100, 100);
std::uniform_real_distribution<double> dist_double(-1.0, 1.0);

// Function to fill integer matrices with random values
void fillMatrix(int *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = dist_int(generator);
    }
}

// Function to fill float matrices with random values
void fillMatrix(float *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = dist_float(generator);
    }
}

// Function to fill double matrices with random values
void fillMatrix(double *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = dist_double(generator);
    }
}

// Template function to run and time matrix multiplication
template <typename T>
void runMatMul(void (*matmulFunc)(const T *, const T *, T *, unsigned int, unsigned int),
               unsigned int n, unsigned int block_dim) {
    T *A = new T[n * n];
    T *B = new T[n * n];
    T *C = new T[n * n];

    fillMatrix(A, n);
    fillMatrix(B, n);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Call the matrix multiplication function
    matmulFunc(A, B, C, n, block_dim);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate time taken
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the first and last elements and time taken
    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << " ms" << std::endl;

    // Free memory
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

    std::cout << "Testing matmul_1 (int)" << std::endl;
    runMatMul(matmul_1, n, block_dim);

    std::cout << "Testing matmul_2 (float)" << std::endl;
    runMatMul(matmul_2, n, block_dim);

    std::cout << "Testing matmul_3 (double)" << std::endl;
    runMatMul(matmul_3, n, block_dim);

    return 0;
}
