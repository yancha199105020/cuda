#include <iostream>
#include <cuda_runtime.h>

/********
 * 求两个数组之和，假如每个数组有1024个元素，就创建1024个线程。每个线程求对应序号的元素之和
 * 如果创建512个线程，那么每个线程就求对应的两个序号之和
 * ***********/
__global__ void add(int *a,int *b,int *c,int lenght){
    int i=blockIdx.x*blockDim.x+threadIdx.x;//blockDim.x:线程块的X轴线程数量，blockIdx.x：线程块的x轴序号
    if(i<lenght){
        c[i]=a[i]+b[i];
    }
}

int main(){
    int numElements =1024;//元素个数
    size_t size =numElements*sizeof(int);//内存长度
    int *a,*b,*c;
    //分配统一内存
    cudaMallocManaged(&a,size);
    cudaMallocManaged(&b,size);
    cudaMallocManaged(&c,size);

    //初始化数据
    for(int i=0;i<numElements;i++){
        a[i]=i;
        b[i]=i;
        c[i]=-1;
    }

    //加载核函数
    int  threadsPerBlock=256;//线程块大小
    int blocksPerGrid=(numElements+threadsPerBlock-1)/threadsPerBlock;//网格大小，保证numElements不能整除threadsPerBlock也能覆盖所有
    add<<<blocksPerGrid,threadsPerBlock>>>(a,b,c,numElements);//调用核函数

    //等待GPU执行完毕
    cudaDeviceSynchronize();
    
    //答应求和结果
    for(int i=0;i<numElements;i++){
        printf("%d ",c[i]);
    }
    printf("\n");

    //释放内存

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}