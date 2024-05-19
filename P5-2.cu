#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
using namespace cv;
using namespace std;

//CPU执行 RGB图像转换为灰度图像
__global__ void rgbToGrayKernel(unsigned char * input, unsigned char *output,int width,int height){
    int x=blockIdx.x*blockDim.x+threadIdx.x;//相当于获取像素点的x坐标地址
    int y=blockIdx.y*blockDim.y+threadIdx.y;//y坐标地址
    if(x<width&&y<height){
        int idx=y*width+x;
        unsigned char b=input[3*idx];//获取像素点b分量
        unsigned char g=input[3*idx+1];//获取像素点g分量
        unsigned char r=input[3*idx+2];//获取像素点r分量
        unsigned char gray=(unsigned char)(0.299f*r+0.587f*g+0.114f*b);
        output[idx]=gray;
    }
}

//CPU执行 RGB图像转换为灰度图像
void rgbToGrayCPU(unsigned char * input, unsigned char *output,int width,int height){
    int lenght=width*height;
    for(int i=0;i<lenght;i++){
        unsigned char b=input[3*i];//获取像素点b分量
        unsigned char g=input[3*i+1];
        unsigned char r=input[3*i+2];
        unsigned char gray=(unsigned char)(0.299f*r+0.587f*g+0.114f*b);
        output[i]=gray;
    }
}

//多次重复计算
__global__ void rgbToGrayMultiple(unsigned char * input, unsigned char *output,int width,int height,int number){
    for(int k=0;k<number;k++){
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
}




int main(){
    VideoCapture cap(0);//打开摄像头0
    if(!cap.isOpened()){
        std::cerr<<"Error opening camera"<<std::endl;
        return -1;
    }
    int k=0;
    Mat  img;
    cap >> img;//获取一帧图像
    if(img.empty()){
        std::cerr<<"recvived empty img frame"<<std::endl;
    }
    imshow("image",img);
    int count=0;
    
    std::cout << "Enter:";
    std::cin >> count;//输入调用次数
    std::cout<<"循环运行次数:"<<count<<std::endl;

    /*测量GPU非统一内存编程多次调用运行时间*/
    //申请内存
    unsigned char *d_input=NULL,*d_output=NULL;
    cudaMalloc(&d_input,img.rows*img.cols*3);
    cudaMalloc(&d_output,img.rows*img.cols);

   //初始化参数
    dim3 blockSize(16,16);
    dim3 gridSize((img.cols+blockSize.x-1)/blockSize.x,(img.rows+blockSize.y-1)/blockSize.y);
    Mat gray(img.rows,img.cols, CV_8UC1);

    auto start = std::chrono::high_resolution_clock::now();  //记录GPU并行转换时间开始时间,
    for(k=0;k<count;k++){ //循环调用
        cudaMemcpy(d_input,img.data,img.rows*img.cols*3,cudaMemcpyHostToDevice);  //从主机复制数据到设备
        rgbToGrayKernel<<<gridSize,blockSize>>>(d_input,d_output,img.cols,img.rows);//调用核函数
        cudaDeviceSynchronize();
        cudaMemcpy(gray.data,d_output,img.rows*img.cols,cudaMemcpyDeviceToHost);  //从设备复制数据到主机
    }
    auto end=std::chrono::high_resolution_clock::now();//记录GPU并行转换时间结束时间,
    std::chrono::duration<double, std::milli> elapsed = end - start; //计算时间差
    std::cout << "GPU非统一内存运行耗时: " << elapsed.count() << " 毫秒" << std::endl; //打印

    /*测量CPU多次调用运行时间*/
    Mat cpuGray(img.rows,img.cols, CV_8UC1);
    start=std::chrono::high_resolution_clock::now();
    for(k=0;k<count;k++){
        rgbToGrayCPU(img.data,cpuGray.data,img.cols,img.rows);        
    }
    end=std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU运行耗时: " << elapsed.count() << " 毫秒" << std::endl;

/*测量GPU统一内存编程多次调用运行时间*/
    // 分配托管内存
    unsigned char *cuda_unified_output;
    unsigned char *cuda_unified_img;
    cudaMallocManaged(&cuda_unified_output, img.rows * img.cols);
    cudaMallocManaged(&cuda_unified_img,img.rows*img.cols*img.channels());


    // 创建与托管内存关联的 cv::Mat
    //cv::Mat img_cuda(height, width, img.type(), d_data);
    Mat unifiedImg(img.rows, img.cols, img.type(), cuda_unified_img);
    Mat unifiedGray(img.rows,img.cols, CV_8UC1,cuda_unified_output);

    cap >>unifiedImg;//获取图像
    start=std::chrono::high_resolution_clock::now();
    for(k=0;k<count;k++){
        rgbToGrayKernel<<<gridSize,blockSize>>>(cuda_unified_img,cuda_unified_output,img.cols,img.rows);
    }
    cudaDeviceSynchronize();
    end=std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "GPU统一内存运行耗时: " << elapsed.count() << " 毫秒" << std::endl;//打印运行时间

    /*测量GPU多次循环计算运行时间*/
    start=std::chrono::high_resolution_clock::now();
    rgbToGrayMultiple<<<gridSize,blockSize>>>(cuda_unified_img,cuda_unified_output,img.cols,img.rows,count);
    cudaDeviceSynchronize();
    end=std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "GPU统一内存多次运行耗时: " << elapsed.count() << " 毫秒" << std::endl;
    
    //显示转化结果
    string str;
    cout<<"是否需要显示图像:";
    cin>>str;

    if((str=="y")||str=="yes"){
        imshow("GPU",gray);
        imshow("CPU",cpuGray);
        imshow("unifiedImg",unifiedImg);
        imshow("unifiedGray",unifiedGray);
        waitKey(0);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(cuda_unified_img);
    cudaFree(cuda_unified_output);
    return 0;
}
