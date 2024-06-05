#include <iostream>
#include <opencv2/opencv.hpp>
#include "P6-sobel.cuh"

int main(int argc, char** argv) {   
    cv::Mat  image;
    cv::Mat grayImage,abs_grad;
    //获取一幅图像
    image = cv::imread("1.png");//原图像image
    // 将彩色图像转换为灰度图像
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);//灰度图像
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat sobelResult(rows, cols, CV_64F);//64 位浮点数 (double)

    // Sobel算子边缘检测
    sobelEdgeDetection(grayImage.ptr<unsigned char>(0), sobelResult.ptr<double>(0), rows, cols);

    // 显示图像
    cv::imshow("Original Image", image);//原图像image
    cv::imshow("Original gray",grayImage);//灰度图像
    cv::imshow("Sobel sobelResult", sobelResult);//边缘检测计算后CV_64F图像
    cv::convertScaleAbs(sobelResult, abs_grad);  //数据转化为CV_8U (unsigned char)
    cv::imshow("Sobel abs_grad", abs_grad);//边缘检测后的CV_8U图像
    
    cv::waitKey(0);
    return 0;
}
