#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include "vscale.cuh"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return -1;
    }

    // Read the size of the array from the command line
    unsigned int n = std::atoi(argv[1]);

    // Allocate host memory for arrays a and b
    float* hA = new float[n];
    float* hB = new float[n];

    // Random number generation for filling arrays a and b
    std::mt19937 generator(759);  // Random number generator with fixed seed
    std::uniform_real_distribution<float> distributionA(-10.0f, 10.0f); // Range for a
    std::uniform_real_distribution<float> distributionB(0.0f, 1.0f);   // Range for b

    // Fill arrays a and b with random values
    for (unsigned int i = 0; i < n; ++i) {
        hA[i] = distributionA(generator);
        hB[i] = distributionB(generator);
    }

    // Allocate device memory
    float *dA, *dB;
    cudaMalloc((void**)&dA, n * sizeof(float));
    cudaMalloc((void**)&dB, n * sizeof(float));

    // Copy host arrays to device
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CUDA event for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel
    int blockSize = 16;  // 16 threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks needed

    // Record the start time
    cudaEventRecord(start);

    // Launch the vscale kernel
    vscale<<<numBlocks, blockSize>>>(dA, dB, n);

    // Wait for the kernel to finish and check for errors
    cudaDeviceSynchronize();

    // Record the stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the time taken for kernel execution
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << std::endl;

    // Copy the result back to host
    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first and last elements of the resulting array
    std::cout << "First element of resulting array: " << hB[0] << std::endl;
    std::cout << "Last element of resulting array: " << hB[n - 1] << std::endl;

    // Free the allocated memory
    delete[] hA;
    delete[] hB;
    cudaFree(dA);
    cudaFree(dB);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
