/**
 * Shawn Zhu
 * convolution.cpp
 * 
 * Credit:
 * 
 */

#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    /**
     * Definitions based on the problem
     * x,y go from 0 to n-1
     * f is image
     * w is mask
     * g is output
     * m is the dimensions of the square matrix
     */

    // Initialize local variables
    int imgX, imgY;
    float pixelVal;
    int offset = (m - 1) / 2;

    // Run through every pixel in image
    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            // This is the output pixel location. Initialize to 0
            output[x * n + y] = 0.0f;

            // Apply mask
            for (int i = 0; i < m; i++){
                for (int j = 0; j < m; j++){
                    // Calculate indicies in f(x + i - (m-1)/2), y + j - (m-1)/2))
                    imgX = x + i - ((m-1) >> 1);
                    imgY = y + j - ((m-1) >> 1);

                    if (imgX >= 0 && imgX < n && imgY >= 0 && imgY < n) {
                        // In bounds: use the image pixel
                        pixelVal = image[imgX * n + imgY];  

                    } 
                    else if ((imgX == -offset && imgY == -offset) || 
                            (imgX == -offset && imgY == n + offset - 1) || 
                            (imgX == n + offset - 1 && imgY == -offset) || 
                            (imgX == n + offset - 1 && imgY == n + offset - 1)) {
                        // Corner padding        
                        pixelVal = 0.0f; 

                    } 
                    else {
                        // Edge padding
                        pixelVal = 1.0f; 

                    }

                    output[x * n + y] += mask[i * m + j] * pixelVal;

                }
            }
        }
    }
}