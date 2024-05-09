#include <stdio.h>
__global__ void helloGPU(){
    printf("this is gpu，blockIdx.x=%d,threadIdx,x=%d\n",blockIdx.x,threadIdx.x);
}
int main(){
    printf("this is cpu\n");
    helloGPU<<<2,3>>>();
    cudaDeviceSynchronize();
    return 0;
}