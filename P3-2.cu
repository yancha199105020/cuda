#include <iostream>
#include <cuda_runtime.h>

// 核函数定义
__global__ void matrixAdd(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;
    if (x < width && y < height) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    // 定义矩阵尺寸
    int width = 256;
    int height = 256;
    int size = width * height * sizeof(float);

    // 分配和初始化主机内存
    float *h_A = new float[width * height];
    float *h_B = new float[width * height];
    float *h_C = new float[width * height];

    // 初始化数组
    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;  // 初始化为 1.0
        h_B[i] = 2.0f;  // 初始化为 2.0
        h_C[i] = -1.0f; //
        if((i%10001)==0){//用于验证求和结果
            h_A[i] = 0.0f;  
            h_B[i] = 0.0f;  
        }
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据从主机到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置执行配置
    dim3 threadsPerBlock(16, 16); // 每个块16x16线程
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 执行核函数
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

        //等待GPU执行完毕
    cudaDeviceSynchronize();
    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < width * height; i++) {
        if (h_C[i] != 3.0f) {
            std::cerr << "add failed at index " << i << std::endl;
        }
    }

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
