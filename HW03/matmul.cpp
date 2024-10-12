/**
 * Shawn Zhu
 * matmul.cpp
 * 
 * Credit: Copied from last HW and modified
 * 
 */
#include "matmul.h"

void mmul(const double* A, const double* B, double* C, const unsigned int n){
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int k = 0; k < n; k++){
            for(unsigned int j = 0; j < n; j++){
                // Switch the order of j and k
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}