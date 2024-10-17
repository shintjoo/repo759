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

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    /** 3 cases. 
     * Size is <= 1 so it doesn't need to be sorted
     * Size is less than the threshold -> serial sort
     * Size is more than the threshold -> parallel sort
     */

    // Case 1: Size is <= 1
    if (n <= 1) {
        return;
    }

    // Case 2: Size is less than threshold
    if (n <= threshold) {
        serial_msort(arr, n);
        return;
    }

    // Divide the array into two halves
    std::size_t mid = n / 2;
    int* left = new int[mid];
    int* right = new int[n - mid];

    std::copy(arr, arr + mid, left);
    std::copy(arr + mid, arr + n, right);

    // Sort both halves in parallel
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                msort(left, mid, threshold);
            }
            #pragma omp task
            {
                msort(right, n - mid, threshold);
            }

            // Wait for both tasks to complete. Needed to synchronize tasks before merging or else merging may be incorrect
            #pragma omp taskwait
        }
    }

    // Merge the sorted halves
    merge(arr, left, mid, right, n - mid);

    delete[] left;
    delete[] right;


}