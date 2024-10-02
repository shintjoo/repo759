/**
 * Shawn Zhu
 * task2.cpp
 * 
 * Credit:
 * 
 */

#include <iostream>
#include <random>
#include <chrono>
#include "convolution.h"

int main(int argc, char** argv) {
    // Variables needed for the Mersenne Twister Enginer
    int some_seed = 759;
    std::mt19937 generator(some_seed);

    // Check if the correct number of arguments is provided
    if (argc != 3) {
        std::cerr << "Error: Incorrect command-line argument." << std::endl;
        return 1;
    }
 
   /*  // Variables used for testing
    const int n = 4;
    const int m = 3;
    float mask[m*m] = {0,0,1,0,1,0,1,0,0};
    float image[n*n] = {1,3,4,8,6,5,2,4,3,4,6,8,1,4,5,2};
    float* output = new float[16];
 */
    // Assign image and mask size
    const size_t n = atoi(argv[1]);
    const size_t m = atoi(argv[2]);
   
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

   /*  // Print the entire output array (Used for testing)
    std::cout << "Output array:" << std::endl;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << output[i * n + j] << " ";
        }
        std::cout << std::endl;
    } */

    // Deallocate memory
    delete[] mask;
    delete[] image;
    delete[] output;

    return 0;
}