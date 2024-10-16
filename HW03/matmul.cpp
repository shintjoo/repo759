/**
 * Shawn Zhu
 * matmul.cpp
 * 
 * Credit: Copied from last HW and modified
 * 
 */
#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const std::size_t n){
#pragma omp parallel for
    for(std::size_t i = 0; i < n; i++){
        for(std::size_t k = 0; k < n; k++){
            for(std::size_t j = 0; j < n; j++){
                // Switch the order of j and k
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}