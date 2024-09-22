/**
 * Shawn Zhu
 * scan.cpp
 * 
 * Credit:
 * 
 */

#include "scan.h"
#include <cstdlib>

float* scanFunction(const float* inputArray, int n) {
    // Allocate memory for the output array
    float* outputArray = (float*)malloc(n * sizeof(float));
    
    // Memory allocation failed
    if (!outputArray) {
        return nullptr;  
    }

    // Perform the inclusive scan
    outputArray[0] = inputArray[0];  // The first element is the same
    for (int i = 1; i < n; i++) {
        // Compute the sum for the output array
        outputArray[i] = outputArray[i - 1] + inputArray[i];
    }

    return outputArray;
}