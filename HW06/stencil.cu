#include <cuda.h>
#include <iostream>
#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R)
{
    // Shared memory
    extern __shared__ float sharedData[];

    float* sharedImage = sharedData;  
    float* sharedMask = sharedData + blockDim.x + 2 * R;

    unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;


    // Load mask into shared memory
    if (tid < 2 * R + 1) {
        sharedMask[tid] = mask[tid];
    }

    // Load image into shared memory
    if (idx < n) {
        sharedImage[tid] = image[idx];
    }
    else {
        sharedImage[threadIdx.x + R] = 1;
    }

    // Deal wit the left halo
    if(tid < R){
        if (idx >= R){
            sharedImage[tid] = image[idx - R];
        }
        else {
            sharedImage[tid] = 1;
        }
    }

    // Deal with the right halo
    if (tid >= blockDim.x - R)
    {
        if (idx + R < n) {
            sharedImage[tid + R + R] = image[idx + R];
        }
        else {
            sharedImage[tid + R + R] = 1;
        }
    }

    // Synchronize threads to ensure all data is loaded into shared memory before computation
    __syncthreads();

    // Do the convolution
    if(idx < n) {
        float result = 0.0f;
        for (int i = -static_cast<int>(R); i <= static_cast<int>(R); i++) {
            result += sharedImage[R + tid + i] * sharedMask[i + R];
        }
        output[tid] = result;
    }
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    // Define num threads and num blocks
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int sharedMem = (threads_per_block + 2 * R) + (2 * R + 1);

    // Launch Kernel
    stencil_kernel<<<blocks, threads_per_block, sharedMem * sizeof(float)>>>(image, mask, output, n, R);
    
    // Synchronize the device
    cudaDeviceSynchronize();
}