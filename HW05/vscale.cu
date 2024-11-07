#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {
    // Calculate the thread index based on block and thread IDs
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that thread index is within bounds
    if (idx < n) {
        b[idx] = a[idx] * b[idx];  // Perform element-wise multiplication
    }
}