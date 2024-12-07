#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include "reduce.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 N threads_per_block" << std::endl;
        return EXIT_FAILURE;
    }

    // Parse command-line arguments
    unsigned int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    if (N <= 0 || threads_per_block <= 0) {
        std::cerr << "N and threads_per_block must be positive integers." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate host memory
    float *h_input = new float[N];

    // Fill the host array with random numbers in the range [-1, 1] using Mersenne Twister
    std::mt19937 mt(std::random_device{}());  // Mersenne Twister RNG
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // Range [-1, 1]

    for (unsigned int i = 0; i < N; ++i) {
        h_input[i] = dist(mt);
    }

    // Allocate device memory for input and output arrays
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    // Calculate the number of blocks needed for the first call to the kernel
    unsigned int num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc((void **)&d_output, num_blocks * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Call the reduce function
    reduce(&d_input, &d_output, N, threads_per_block);

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to the host
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting sum
    std::cout << result << std::endl;

    // Print the time taken in milliseconds
    std::cout << milliseconds << "ms" << std::endl;

    // Free host and device memory
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
