/**
 * Shawn Zhu
 * msort.cpp
 * 
 * Credit: 
 * 
 */
#include "msort.h"
#include <algorithm>

// Function to merge sorted halves
void merge(int* arr, int* left, std::size_t left_size, int* right, std::size_t right_size) {
    std::size_t i = 0, j = 0, k = 0;

    // Determines whether the value in the left half or the right half should be added
    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    // Copy any remaining elements from the left half
    while (i < left_size) {
        arr[k++] = left[i++];
    }

    // Copy any remaining elements from the right half
    while (j < right_size) {
        arr[k++] = right[j++];
    }
}

void serial_msort(int* arr, std::size_t n) {
    if (n <= 1) {
        return;
    }

    // Divide the array into two halves
    std::size_t mid = n / 2;
    int* left = new int[mid];
    int* right = new int[n - mid];

    std::copy(arr, arr + mid, left);
    std::copy(arr + mid, arr + n, right);

    // Recursively sort both halves
    serial_msort(left, mid);
    serial_msort(right, n - mid);

    // Merge the sorted halves
    merge(arr, left, mid, right, n - mid);

    delete[] left;
    delete[] right;

}

// Parallel merge sort using OpenMP tasks
void parallel_msort(int* arr, std::size_t n, std::size_t threshold) {
    if (n <= 1) {
        return; 
    }

    if (n <= threshold) {
        serial_msort(arr, n);
    } else {
        std::size_t mid = n / 2;
        int* left = new int[mid];
        int* right = new int[n - mid];

        std::copy(arr, arr + mid, left);
        std::copy(arr + mid, arr + n, right);

        // Use OpenMP tasks for parallel recursive sorting
        #pragma omp task shared(left, mid, threshold)
        parallel_msort(left, mid, threshold);

        #pragma omp task shared(right, n, mid, threshold)
        parallel_msort(right, n - mid, threshold);

        // Wait for both tasks to complete
        #pragma omp taskwait

        merge(arr, left, mid, right, n - mid);

        delete[] left;
        delete[] right;
    }
}


void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_msort(arr, n, threshold);
        }
    }
}