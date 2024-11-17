#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, int R)
{
    // Shared memory
    extern __shared__ float sharedData[];

    float* sharedImage = &sharedData[0];  
    float* sharedMask = &sharedData[blockDim.x + 2 * R];

    unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= n) return;

    // Load mask into shared memory
    if (tid < 2 * R + 1) {
        sharedMask[tid] = mask[tid];
    }

    // Load image into shared memory
    if (idx < n) {
        sharedImage[tid] = image[idx];
    }

    // Synchronize threads to ensure all data is loaded into shared memory before computation
    __syncthreads();

    //initialize output values
    float result = 0.0f;
    for (int j = -R; j <= R; ++j) {
        int idx = threadIdx.x + j;
        if (idx >= 0 && idx < blockDim.x) {
            result += sharedImage[idx] * sharedMask[j + R];
        }
    }

    // Store the result in global memory (output array)
    if (idx < n) {
        output[idx] = result;
    }
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int sharedCount = (R * 2 + 1) + threads_per_block * 2;
    stencil_kernel<<<blocks, threads_per_block, sharedCount * sizeof(float)>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}