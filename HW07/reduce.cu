#include <cuda.h>
#include <iostream>
#include "reduce.cuh"
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    // Dynamically allocated shared memory
    extern __shared__ float sdata[];

    // Thread ID within the block
    unsigned int tid = threadIdx.x;
    // Global index for the current thread
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    //Load data into shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        sdata[tid] += g_idata[i + blockDim.x];
    }
    __syncthreads();

    // Perform the reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block's reduction to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }

}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    float *d_in = *input;
    float *d_out = *output;

    while (blocks > 1) {
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_in, d_out, N);
        cudaDeviceSynchronize();

        // Update input and output pointers for the next iteration
        N = blocks;
        d_in = d_out;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }

    // Final kernel call
    reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

}