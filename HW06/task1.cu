#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    // Deal with input values
    if (argc != 3) {
        std::cerr << "Wrong inputs";
        return -1;
    }
    size_t n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    // Create matrices A and B on the host
    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n);

    // Fill matrices A and B with random numbers in the range [-1, 1]
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n * n; ++i) {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call matmul function
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Copy result matrix C back to host
    cudaMemcpy(C.data(), d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of matrix C
    std::cout << C[n * n - 1] << std::endl;

    // Print elapsed time
    std::cout << elapsed_time_ms << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

