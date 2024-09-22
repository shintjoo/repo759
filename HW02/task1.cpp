/**
 * Shawn Zhu
 * task1.cpp
 * 
 * Credit:
 * 
 */

#include <iostream>
#include <chrono>
#include "scan.h"

int main(int argc, char* argv[]) {
    // Local variable declaration
    int n;
    float* inputArray;
    float* outputArray;

    // Take in n as the first command line arguement
    // Check if the correct number of arguments is provided
    if (argc < 2) {
        std::cerr << "Error: Please provide an integer n as a command-line argument." << std::endl;
        return 1;
    }

    // Get the integer that is given in the command line
    n = std::atoi(argv[1]);

    // Fill the array with n rand float values between -1.0 and 1.0
    inputArray = (float*)malloc(n * sizeof(float));
    if (!inputArray) {
        std::cerr << "Malloc failed for inputArray.\n";
        return 1;
    }

    srand(time(0));
    for (int i = 0; i < n; i++) {
        inputArray[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / 2) - 1.0f;
    }

    // Start Timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run scan function
    outputArray = scanFunction(inputArray, n);

    // Stop Timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);

    // This needs to be outside the timer as to not affect the duration
    if (!outputArray) {
        std::cerr << "Malloc failed for outputArray.\n";
        return 1;
    }

    // Print out time taken for scan
    std::cout << "Runtime: " << duration.count() << " milliseconds"<< std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "Input " << inputArray[i] << std::endl;
        std::cout << "Output " << outputArray[i] << std::endl;
    }
    //std::cout << "Last element: " << outputArray[n - 1] << std::endl;

    // Free allocated memory
    free(inputArray);
    free(outputArray);

    return 0;
}