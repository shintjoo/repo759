#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

// Kernel for interger matmul
__global__ void tiledMatMulKernelInt(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    extern __shared__ int shared_memory_int[];
    int *tile_A = &shared_memory_int[0];
    int *tile_B = &shared_memory_int[block_dim * block_dim];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * block_dim + ty;
    unsigned int col = blockIdx.x * block_dim + tx;

    int value = 0;

    for (unsigned int t = 0; t < (n + block_dim - 1) / block_dim; ++t) {
        if (row < n && t * block_dim + tx < n)
            tile_A[ty * block_dim + tx] = A[row * n + t * block_dim + tx];
        else
            tile_A[ty * block_dim + tx] = 0;

        if (col < n && t * block_dim + ty < n)
            tile_B[ty * block_dim + tx] = B[(t * block_dim + ty) * n + col];
        else
            tile_B[ty * block_dim + tx] = 0;

        __syncthreads();

        for (unsigned int i = 0; i < block_dim; ++i) {
            value += tile_A[ty * block_dim + i] * tile_B[i * block_dim + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// Kernel for float matmul
__global__ void tiledMatMulKernelFloat(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    extern __shared__ float shared_memory_float[];
    float *tile_A = &shared_memory_float[0];
    float *tile_B = &shared_memory_float[block_dim * block_dim];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * block_dim + ty;
    unsigned int col = blockIdx.x * block_dim + tx;

    float value = 0;

    for (unsigned int t = 0; t < (n + block_dim - 1) / block_dim; ++t) {
        if (row < n && t * block_dim + tx < n)
            tile_A[ty * block_dim + tx] = A[row * n + t * block_dim + tx];
        else
            tile_A[ty * block_dim + tx] = 0.0f;

        if (col < n && t * block_dim + ty < n)
            tile_B[ty * block_dim + tx] = B[(t * block_dim + ty) * n + col];
        else
            tile_B[ty * block_dim + tx] = 0.0f;

        __syncthreads();

        for (unsigned int i = 0; i < block_dim; ++i) {
            value += tile_A[ty * block_dim + i] * tile_B[i * block_dim + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

// Kernel for double matmul
__global__ void tiledMatMulKernelDouble(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    extern __shared__ double shared_memory_double[];
    double *tile_A = &shared_memory_double[0];
    double *tile_B = &shared_memory_double[block_dim * block_dim];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * block_dim + ty;
    unsigned int col = blockIdx.x * block_dim + tx;

    double value = 0;

    for (unsigned int t = 0; t < (n + block_dim - 1) / block_dim; ++t) {
        if (row < n && t * block_dim + tx < n)
            tile_A[ty * block_dim + tx] = A[row * n + t * block_dim + tx];
        else
            tile_A[ty * block_dim + tx] = 0.0;

        if (col < n && t * block_dim + ty < n)
            tile_B[ty * block_dim + tx] = B[(t * block_dim + ty) * n + col];
        else
            tile_B[ty * block_dim + tx] = 0.0;

        __syncthreads();

        for (unsigned int i = 0; i < block_dim; ++i) {
            value += tile_A[ty * block_dim + i] * tile_B[i * block_dim + tx];
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

    tiledMatMulKernelInt<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);

    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    int blockNum = (n + block_dim - 1) / block_dim;

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    tiledMatMulKernelFloat<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);

    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C,unsigned int n, unsigned int block_dim){
    int blockNum = (n + block_dim - 1) / block_dim;

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(blockNum, blockNum);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);

    tiledMatMulKernelDouble<<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);

    cudaDeviceSynchronize();
}