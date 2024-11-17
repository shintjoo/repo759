#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    // Calculate the index for the element this thread produces the result for
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check that the index is within the size of the resulting array
    if (idx < n * n) {
        size_t row = idx / n;
        size_t col = idx % n;
        float value = 0.0f;

        // Perform dot product of the row of A with the column of B
        for (size_t k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }

        // Store the result in C
        C[row * n + col] = value;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){

    // Define num threads and num blocks
    size_t total_elements = n * n;
    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);

    // Synchronize the device
    cudaDeviceSynchronize();
}