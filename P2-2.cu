#include <stdio.h>
__global__ void helloGPU(){
    printf("this is gpu，blockIdx.x=%d,blockIdx.y=%d，threadIdx.x=%d,threadIdx.y=%d\n",\
	blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
}
int main(){
    printf("this is cpu\n");
	dim3 blocksize(2,2);
	dim3 gridsize(2,3);
    helloGPU<<<gridsize,blocksize>>>();
    cudaDeviceSynchronize();
    return 0;
}
