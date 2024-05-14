#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
using namespace cv;

__global__ void rgbToGrayKernel(unsigned char * input, unsigned char *output,int width,int height){
    int x=blockIdx.x*blockDim.x+threadIdx.x;//相当于获取像素点的x坐标地址
    int y=blockIdx.y*blockDim.y+threadIdx.y;//y坐标地址
    if(x<width&&y<height){
        int idx=y*width+x;
        unsigned char b=input[3*idx];//获取像素点b分量
        unsigned char g=input[3*idx+1];
        unsigned char r=input[3*idx+2];
        unsigned char gray=(unsigned char)(0.299f*r+0.587f*g+0.114f*b);
        output[idx]=gray;
    }
}

int main(){
    VideoCapture cap(0);//打开摄像头0
    if(!cap.isOpened()){
        std::cerr<<"Error opening camera"<<std::endl;
        return -1;
    }
    Mat  img;
    cap >> img;
    if(img.empty()){
        std::cerr<<"recvived empty img frame"<<std::endl;
    }
    imshow("image",img);
    
    //申请内存
    unsigned char *d_input=NULL,*d_output=NULL;
    cudaMalloc(&d_input,img.rows*img.cols*3);
    cudaMalloc(&d_output,img.rows*img.cols);

    //从主机复制数据到设备
    cudaMemcpy(d_input,img.data,img.rows*img.cols*3,cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((img.cols+blockSize.x-1)/blockSize.x,(img.rows+blockSize.y-1)/blockSize.y);

    rgbToGrayKernel<<<gridSize,blockSize>>>(d_input,d_output,img.cols,img.rows);
    cudaDeviceSynchronize();

    Mat gray(img.rows,img.cols, CV_8UC1);

    //从设备复制数据到主机
    cudaMemcpy(gray.data,d_output,img.rows*img.cols,cudaMemcpyDeviceToHost);
    //释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    //显示转化结果
    imshow("gray",gray);
    waitKey(0);
    return 0;
}