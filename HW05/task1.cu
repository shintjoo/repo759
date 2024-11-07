#include <cstdio>

__global__ void computeFactorial() {
    int n = threadIdx.x + 1; // Threads 1 to 8 compute factorials for numbers 1 to 8
    int result = 1;
    
    // Calculate factorial directly in the kernel
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }

    // Print in the required format
    printf("%d!=%d\n", n, result);
}

int main() {
    // Launch the kernel with 1 block and 8 threads
    computeFactorial<<<1, 8>>>();

    // Synchronize to ensure all threads complete before the host returns
    cudaDeviceSynchronize();

    return 0;
}
