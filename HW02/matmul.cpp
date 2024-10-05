/**
 * Shawn Zhu
 * matmul.cpp
 * 
 * Credit: Completed on my own
 * 
 */
#include "matmul.h"

/**
 * index i through the rows of C
 * index j through the columns of C
 * index k through; i.e., to carry out, the dot product of the ith row A with the jth column of B.
 */
void mmul1(const double* A, const double* B, double* C, const unsigned int n){
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            for(unsigned int k = 0; k < n; k++){
                // Row major order: Array[row*(# of columns)+column]
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int k = 0; k < n; k++){
            for(unsigned int j = 0; j < n; j++){
                // Switch the order of j and k
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n){ 
    for(unsigned int j = 0; j < n; j++){
        for(unsigned int k = 0; k < n; k++){
            for(unsigned int i = 0; i < n; i++){
                // move i to be the innermost loop
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n){
    for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < n; j++){
            for(unsigned int k = 0; k < n; k++){
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}
