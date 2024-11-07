#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random> 
#include <cuda_runtime.h>

__global__ void compute(int *dA, int a) {
    // Each thread computes ax + y
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < 16){
        dA[index] = a * threadIdx.x + blockIdx.x;
    }
}

int main() {
    const int arraySize = 16;
    int hA[arraySize]; // Host array to store results
    int *dA; // Device array
    
    // Random number generation setup
    int some_seed = 759;
    std::mt19937 generator(some_seed); 
    std::uniform_int_distribution<int> distribution(1, 10); 
    int a = distribution(generator);
    
    // Allocate memory on the device
    cudaMalloc((void**)&dA, arraySize * sizeof(int));

    // Launch the kernel
    compute<<<2, 8>>>(dA, a);

    // Check for any errors in kernel launch
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the data from device to host
    cudaMemcpy(hA, dA, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result array
    std::cout << "Results: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << hA[i] << " ";
    }
    std::cout << std::endl;

    // Free the device memory
    cudaFree(dA);

    return 0;
}
