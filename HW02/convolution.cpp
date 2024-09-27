/**
 * Shawn Zhu
 * convolution.cpp
 * 
 * Credit:
 * 
 */

#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){

    // Definitions based on the problem
    // x,y go from 0 to n-1
    // f is image
    // w is mask
    // g is output
    // m is the dimensions of the square matrix

    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            for (int i = 0; i < m; i++){
                for (int j = 0; j < m; j++){
                    
                }
            }
        }
    }

}