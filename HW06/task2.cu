#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda.h>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    // Deal with input values
    if (argc != 4) {
        std::cerr << "Wrong inputs";
        return -1;
    }
    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    // Initialize arrays for host
    float* h_image = new float[n];
    float* h_mask = new float[2 * R + 1];
    float* h_output = new float[n];

    // Fill matrices A and B with random numbers in the range [-1, 1]
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < n; i++) {
        h_image[i] = dist(generator);
    }
    for (unsigned int i = 0; i < 2 * R + 1; i++) {
        h_mask[i] = dist(generator);
    }

    // Allocate device memory
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call stencil function
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Copy result matrix back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of matrix h_output
    std::cout << h_output[n - 1] << std::endl;

    // Print elapsed time
    std::cout << elapsed_time_ms << std::endl;

    // Free host memory
    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;

    //Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}