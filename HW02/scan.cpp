/**
 * Shawn Zhu
 * scan.cpp
 * 
 * Credit: Completed alone
 * 
 */

#include "scan.h"

void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) {
        return; // No elements to process
    }

    output[0] = arr[0]; // The first element is the same
    for (std::size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + arr[i]; // Perform the scan
    }
}