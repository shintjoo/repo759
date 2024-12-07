#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

// Kernel for interger matmul
__global__ void tiledMatMulKernelInt(const int *A, const int *B, int *C, unsigned int n) {
    extern __shared__ int shared_memory_int[];
    int *tile_A = &shared_memory_int[0];
    int *tile_B = &shared_memory_int[blockDim.y * blockDim.x];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * blockDim.y + ty;
    unsigned int col = blockIdx.x * blockDim.x + tx;

    int value = 0;

    for (unsigned int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t) {
        if (row < n && t * blockDim.x + tx < n)
            tile_A[ty * blockDim.x + tx] = A[row * n + t * blockDim.x + tx];
        else
            tile_A[ty * blockDim.x + tx] = 0;

        if (col < n && t * blockDim.y + ty < n)
            tile_B[ty * blockDim.x  + tx] = B[(t * blockDim.x  + ty) * n + col];
        else
            tile_B[ty * blockDim.x  + tx] = 0;

        __syncthreads();

        for (unsigned int i = 0; i < blockDim.x ; ++i) {
            value += tile_A[ty * blockDim.x + i] * tile_B[i * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// Kernel for float matmul
__global__ void tiledMatMulKernelFloat(const float *A, const float *B, float *C, unsigned int n) {
    extern __shared__ float shared_memory_float[];
    float *tile_A = &shared_memory_float[0];
    float *tile_B = &shared_memory_float[blockDim.y * blockDim.x];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * blockDim.y + ty;
    unsigned int col = blockIdx.x * blockDim.x + tx;

    float value = 0;

    for (unsigned int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t) {
        if (row < n && t * blockDim.x + tx < n)
            tile_A[ty * blockDim.x + tx] = A[row * n + t * blockDim.x + tx];
        else
            tile_A[ty * blockDim.x + tx] = 0.0f;

        if (col < n && t * blockDim.y + ty < n)
            tile_B[ty * blockDim.x + tx] = B[(t * blockDim.x + ty) * n + col];
        else
            tile_B[ty * blockDim.x + tx] = 0.0f;

        __syncthreads();

        for (unsigned int i = 0; i < blockDim.x; ++i) {
            value += tile_A[ty * blockDim.x + i] * tile_B[i * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// Kernel for double matmul
__global__ void tiledMatMulKernelDouble(const double *A, const double *B, double *C, unsigned int n) {
    extern __shared__ double shared_memory_double[];
    double *tile_A = &shared_memory_double[0];
    double *tile_B = &shared_memory_double[blockDim.y * blockDim.x];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * blockDim.y + ty;
    unsigned int col = blockIdx.x * blockDim.x + tx;

    double value = 0;

    for (unsigned int t = 0; t < (n + blockDim.x - 1) / blockDim.x; ++t) {
        if (row < n && t * blockDim.x + tx < n)
            tile_A[ty * blockDim.x + tx] = A[row * n + t * blockDim.x + tx];
        else
            tile_A[ty * blockDim.x + tx] = 0.0;

        if (col < n && t * blockDim.y + ty < n)
            tile_B[ty * blockDim.x + tx] = B[(t * blockDim.x + ty) * n + col];
        else
            tile_B[ty * blockDim.x + tx] = 0.0;

        __syncthreads();

        for (unsigned int i = 0; i < blockDim.x; ++i) {
            value += tile_A[ty * blockDim.x + i] * tile_B[i * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// Host for int
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    int blockNum = (n + block_dim - 1) / block_dim;

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);

    tiledMatMulKernelInt<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    int blockNum = (n + block_dim - 1) / block_dim;

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    tiledMatMulKernelFloat<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C,unsigned int n, unsigned int block_dim){
    int blockNum = (n + block_dim - 1) / block_dim;

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);

    tiledMatMulKernelDouble<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}