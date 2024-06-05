#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <cuda_runtime.h>

void sobelEdgeDetection(unsigned char* inputImage, double* outputImage, int rows, int cols);

#endif
