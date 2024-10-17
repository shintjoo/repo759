/**
 * Shawn Zhu
 * task2.cpp
 * 
 * Credit: Copied from HW02
 * 
 */

#include <iostream>
#include <random>
#include <chrono>
#include "convolution.h"
#include <omp.h>

int main(int argc, char** argv) {
    // Variables needed for the Mersenne Twister Enginer
    int some_seed = 759;
    std::mt19937 generator(some_seed);

    // Check if the correct number of arguments is provided
    if (argc < 2) {
        std::cerr << "Error: Incorrect command-line argument." << std::endl;
        return 1;
    }

    // Assign image and mask size
    std::size_t n = atoi(argv[1]);
    std::size_t m = 3;

    // Assign the second input variable to be the number of threads
    omp_set_num_threads(std::atoi(argv[2]));
    
    // Initialize the mask and image arrays
    float* mask = new float[m*m];
    float* image = new float[n*n];
    float* output = new float[n*n];

    // Setting distibution for mask and image
    std::uniform_real_distribution<float> dist_mask(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist_image(-10.0, 10.0); // uniform distribution between -10.0 and 10.0

    for (std::size_t i = 0; i < m*m; i++) {
        mask[i] = dist_mask(generator);
    }

    for (std::size_t j = 0; j < n*n; j++) {
        image[j] = dist_image(generator);
    }

    // Start Timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run convolve function
    convolve(image, output, n, mask, m);

    // Stop Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);

    // Print Output
    std::cout << duration.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[(n*n) - 1] << std::endl;

    // Deallocate memory
    delete[] mask;
    delete[] image;
    delete[] output;

    return 0;
}