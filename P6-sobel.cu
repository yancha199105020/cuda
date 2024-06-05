#include "P6-sobel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FILTER_WIDTH 3
__constant__ int sobelX[FILTER_WIDTH * FILTER_WIDTH];
__constant__ int sobelY[FILTER_WIDTH * FILTER_WIDTH];

__global__ void sobelKernel(unsigned char* input, double* output, int rows, int cols) {
    //将像素点和线程对应，(x,y)为像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    double gx = 0.0, gy = 0.0;
    //卷积计算
    for (int filterY = -1; filterY <= 1; filterY++) {
        for (int filterX = -1; filterX <= 1; filterX++) {
            //min和max并限定取值范围，（x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x,y),(x+1,y)。。。。
            int imgX = min(max(x + filterX, 0), cols - 1);
            int imgY = min(max(y + filterY, 0), rows - 1);
            gx += input[imgY * cols + imgX] * sobelX[(filterY + 1) * FILTER_WIDTH + (filterX + 1)];//X方向偏导数
            gy += input[imgY * cols + imgX] * sobelY[(filterY + 1) * FILTER_WIDTH + (filterX + 1)];//Y方向偏导数
        }
    }
    //求梯度
    output[y * cols + x] = sqrt(gx * gx + gy * gy);
}

void sobelEdgeDetection(unsigned char* inputImage, double* outputImage, int rows, int cols) {
    unsigned char* d_input;
    double* d_output;

    // Sobel kernel for X and Y directions
    int h_sobelX[FILTER_WIDTH * FILTER_WIDTH] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    int h_sobelY[FILTER_WIDTH * FILTER_WIDTH] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };

    cudaMemcpyToSymbol(sobelX, h_sobelX, sizeof(int) * FILTER_WIDTH * FILTER_WIDTH);
    cudaMemcpyToSymbol(sobelY, h_sobelY, sizeof(int) * FILTER_WIDTH * FILTER_WIDTH);

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(unsigned char));
    cudaMalloc(&d_output, rows * cols * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_input, inputImage, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    sobelKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);

    // Copy the result back to the host
    cudaMemcpy(outputImage, d_output, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
