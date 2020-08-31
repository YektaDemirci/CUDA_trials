/*
  Example code for displaying gstreamer video from the CSI port of the Nvidia Jetson in OpenCV.
  Created by Peter Moran on 7/29/17.
  https://gist.github.com/peter-moran/742998d893cd013edf6d0c86cc86ff7f
*/

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <chrono>
#include <string>
#include <iostream> 
//#include <omp.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#ifndef __SOBELFILTER_KERNELS_H_
#define __SOBELFILTER_KERNELS_H_

// global determines which filter to invoke


#endif

// Enable Nvidia Profiler code in compiling
//#define BUILD_NV_PROFILER

#define UNIFIED_MEMORY_ON

#define DISPLAY_MODE_NO_FILTER 0
#define DISPLAY_MODE_GRAY 1
#define DISPLAY_MODE_GAUSSIAN_BLUR 2
#define DISPLAY_MODE_CANNY 3
#define DISPLAY_MODE_CANNY_WITH_COLOR 4



extern "C" void sobelFilter(cv::Mat& cur_f, int iw, int ih, float fScale,cv::Mat& out_f);

void gray(cv::cuda::GpuMat& color_in, cv::cuda::GpuMat& gray_out);

extern void combine(cv::cuda::GpuMat& color_in_out, cv::cuda::GpuMat& gray_in);

void display_help();

std::string get_tegra_pipeline(int width, int height, int fps) {

//return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)"+std::to_string(width)+", height=(int)"+std::to_string(height)+",format=(string)I420, framerate=(fraction)"+std::to_string(fps)+"/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink ";

return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458,format=(string)I420, framerate=(fraction)"+std::to_string(fps)+"/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)"+std::to_string(width)+", height=(int)"+std::to_string(height)+", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink ";
//return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)"+std::to_string(width)+", height=(int)"+std::to_string(height)+",format=(string)I420, framerate=(fraction)"+std::to_string(fps)+"/1 ! nvivafilter customer-lib-name=libnvsample_cudaprocess.so cuda-process=true ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvegltransform ! nveglglessink ";

}

//extern cv::Mat dev_q(cv::Mat& avg_f);

const char* params
    = "{ help           | false | print usage         }"
      "{ heyDude        | 40    | model configuration }";

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv, params);
    

    
    // Options2592x1458
    int WIDTH = 1280;//640;
    int HEIGHT = 720;//360./640.*WIDTH;
    int FPS = 30;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Connection failed"<<std::endl;
        return -1;
    }

    #ifdef UNIFIED_MEMORY_ON
    // View video
    // Create Shared Memory between Host and Device
    cv::cuda::HostMem shared_frame(HEIGHT, WIDTH, CV_8UC3, cv::cuda::HostMem::SHARED);
    // Create a header for the host to access the shared memory
    cv::Mat avgframe = shared_frame.createMatHeader();
    // Create a header for Device to access shared memory
    cv::cuda::GpuMat d_img_color = shared_frame.createGpuMatHeader();
    
    #else
        cv::cuda::GpuMat d_img_color(HEIGHT,WIDTH,CV_8UC3);
    
        cv::Mat avgframe(HEIGHT,WIDTH, CV_8UC3);
    #endif
//    cv::Mat gray_edge_det_f(HEIGHT,WIDTH,CV_8UC1);
//    cv::Mat gray_out(HEIGHT,WIDTH,CV_8UC1);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end   = std::chrono::high_resolution_clock::now();
    auto t_elaps = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start);
    float cur_fps = 0;
    int count = 0;
    int blur_size = 5;
    float blur_sigma = 1.7f;
    float small_edge_thresh = 60.0f;
    float large_edge_thresh = 80.0f;
    bool App_state_exit = false;
    int display_mode = DISPLAY_MODE_NO_FILTER;
    bool new_window_name_exists = false;
    
    int key_pressed = -2;
    //float brightness = 1.0f;
    
    //static int imWidth  = WIDTH;   // Image width
    //static int imHeight = HEIGHT;   // Image height
    
    std::string old_window_name="Display window";
    std::string new_window_name;
    std::string display_mode_string;
    
    cv::namedWindow(old_window_name, cv::WINDOW_OPENGL);
    cv::resizeWindow(old_window_name, WIDTH, HEIGHT);
    

    cv::cuda::GpuMat d_img_gray(HEIGHT,WIDTH,CV_8UC1);
    cv::cuda::GpuMat d_img_gray_blurred(HEIGHT,WIDTH,CV_8UC1);
    cv::cuda::GpuMat d_edges(HEIGHT,WIDTH,CV_8UC1);
    /*cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(75.0, 220.0);
    //cv::Ptr<cv::cuda::Filter> median = cv::cuda::createMedianFilter(CV_8UC1, blur_size);
    cv::Ptr<cv::cuda::Filter> gaussian = cv::cuda::createGaussianFilter(CV_8UC1,CV_8UC1,cv::Size(5,5),3,3);*/



    
    display_help();
    
    
    //cv::cuda::setGlDevice(0);
    //cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
    
    #ifdef BUILD_NV_PROFILER // If Nvidia Profiler is enabled for compile
        cudaProfilerInitialize("profcfg.txt","out.txt",cudaKeyValuePair);

        cudaProfilerStart();
    #endif
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(small_edge_thresh, large_edge_thresh);
  
        
    printf("Ehehehe test 1 \n");
    
    
    while(!App_state_exit) 
    //for (int i =0;i<160;i++)
    {
        
        //cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(small_edge_thresh, large_edge_thresh);
        //cv::Ptr<cv::cuda::Filter> median = cv::cuda::createMedianFilter(CV_8UC1, blur_size);
        cv::Ptr<cv::cuda::Filter> gaussian = cv::cuda::createGaussianFilter(CV_8UC1,CV_8UC1,cv::Size(blur_size,blur_size),blur_sigma,blur_sigma);
        
        cap >> avgframe;
        //std::cout<<avgframe.cols<<"x"<<avgframe.rows<<std::endl;
        //sobelFilter(avgframe, imWidth, imHeight, brightness,gray_edge_det_f);
        # ifndef UNIFIED_MEMORY_ON
            d_img_color.upload(avgframe);
            printf("This is used \n");
        # endif
        

        if(display_mode>=DISPLAY_MODE_GRAY)
            gray(d_img_color, d_img_gray);
            

        if(display_mode>=DISPLAY_MODE_GAUSSIAN_BLUR)
            gaussian->apply(d_img_gray, d_img_gray_blurred);
        gaussian.release();
        //median->apply(d_img_gray, d_img_gray_blurred);
        //cv::Canny(gray_out,gray_edge_det_f,20,250);
        
        //printf("Control point 3 \n");
        //cv::cuda::GpuMat d_img(gray_out);

        
        if(display_mode>=DISPLAY_MODE_CANNY)
            canny->detect(d_img_gray_blurred, d_edges);
            //d_img_gray_blurred.copyTo(d_edges);
        
        //printf("Control point 4 \n");
        //d_edges.download(gray_edge_det_f);
        if(display_mode>=DISPLAY_MODE_CANNY_WITH_COLOR)
            combine(d_img_color,d_edges);
        
        
        //std::cout << *gray_edge_det_f.data << std::endl;

        // If 1 seconds has elapsed since last time, calculate FPS
        t_end   = std::chrono::high_resolution_clock::now();
        t_elaps = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start);
        if(t_elaps.count()>=1000000.0f)
        {
        
            // Calculate Average FPS over the last second
            
            
            t_start = std::chrono::high_resolution_clock::now();
            cur_fps = 1000000.0f/(t_elaps.count()/(float)count);

            // Frame counter between
            count = 0;
            new_window_name_exists = true;
            /*std::cout<<"-------------------------------"<<std::endl;
            std::cout<<d_img_color.cols         <<"x"<<d_img_color.rows         <<" "<<d_img_color.step         <<std::endl;
            std::cout<<d_img_gray.cols          <<"x"<<d_img_gray.rows          <<" "<<d_img_gray.step          <<std::endl;
            std::cout<<d_img_gray_blurred.cols  <<"x"<<d_img_gray_blurred.rows  <<" "<<d_img_gray_blurred.step  <<std::endl;
            std::cout<<d_edges.cols             <<"x"<<d_edges.rows             <<" "<<d_edges.step             <<std::endl;*/

        }
        else
        {
            count++;
        }
       
        

        

        //Display the image frame determined by the display_mode
            // Also setup window title accordingly
        switch(display_mode)
        {
            case DISPLAY_MODE_NO_FILTER:
                cv::imshow(old_window_name, d_img_color);
                display_mode_string = "No Filter                |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            case DISPLAY_MODE_GRAY:
                cv::imshow(old_window_name, d_img_gray);
                display_mode_string = "Gray Image               |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            case DISPLAY_MODE_GAUSSIAN_BLUR:
                cv::imshow(old_window_name, d_img_gray_blurred);
                display_mode_string = "Gray Blurred Image       |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps)+" | Blur: Size="+std::to_string(blur_size) +" Sigma="+std::to_string(blur_sigma);
                break;
            case DISPLAY_MODE_CANNY:
                cv::imshow(old_window_name, d_edges);
                display_mode_string = "Canny Detection          |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps)+" | Blur: Size="+std::to_string(blur_size) +" Sigma="+std::to_string(blur_sigma)+" | EDGE: Lower="+std::to_string((int)small_edge_thresh)+" Upper="+std::to_string((int)large_edge_thresh);
                break;
            case DISPLAY_MODE_CANNY_WITH_COLOR:
                cv::imshow(old_window_name, d_img_color);
                display_mode_string = "Canny Detection w/ Color |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps)+" | Blur: Size="+std::to_string(blur_size) +" Sigma="+std::to_string(blur_sigma)+" | EDGE: Lower="+std::to_string((int)small_edge_thresh)+" Upper="+std::to_string((int)large_edge_thresh);
                break;
            default:
                cv::imshow(old_window_name, d_img_color);
                display_mode_string = "No Filter                |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
        }
        
        key_pressed=cv::waitKey(1);


        // Register user key inputs on the display screen and process them
        switch(key_pressed)
        {
            // Change Blur Sigma
            case 43: // '+'
                blur_sigma+=0.1f;
                new_window_name_exists = true;
                break;
            case 45: // '-'
                blur_sigma-=0.1f;
                if(blur_sigma<0.0f) blur_sigma=0.0f;
                else new_window_name_exists = true;
                break;

            // Change Blur Window Size
            case 52: // '4'
                blur_size+=2;
                if(blur_size>29) blur_size=29;
                else new_window_name_exists = true;
                break;
            case 49: // '1'
                blur_size-=2;
                if(blur_size<3) blur_size=3;
                else new_window_name_exists = true;
                break;

            // Change Lower Threshold for Edge Detection
            case 53: // '5'
                small_edge_thresh+=1.0f;
                if(small_edge_thresh>large_edge_thresh) small_edge_thresh=large_edge_thresh;
                else {new_window_name_exists = true;canny->setLowThreshold(small_edge_thresh);}
                break;
            case 50: // '2'
                small_edge_thresh-=1.0f;
                if(small_edge_thresh<0) small_edge_thresh=0.0f;
                else {new_window_name_exists = true;canny->setLowThreshold(small_edge_thresh);}
                break;

            // Change Upper Threshold for Edge Detection
            case 54: // '6'
                large_edge_thresh+=1.0f;
                new_window_name_exists = true;
                canny->setHighThreshold(large_edge_thresh);
                break;
            case 51: // '3'
                large_edge_thresh-=1.0f;
                if(small_edge_thresh>large_edge_thresh) large_edge_thresh= small_edge_thresh;
                else {new_window_name_exists = true;canny->setHighThreshold(large_edge_thresh);}
                break;
            // Quit
            case 27: // 'Esc'
                App_state_exit = true;
                break;
            case 104: // 'h'
                display_help();
                break;
            case 97:
                display_mode=DISPLAY_MODE_NO_FILTER;
                new_window_name_exists = true;
                break;
            case 115:
                display_mode=DISPLAY_MODE_GRAY;
                new_window_name_exists = true;
                break; 
            case 100:
                display_mode=DISPLAY_MODE_GAUSSIAN_BLUR;
                new_window_name_exists = true;
                break;  
            case 102:
                display_mode=DISPLAY_MODE_CANNY;
                new_window_name_exists = true;
                break;    
            case 103:
                display_mode=DISPLAY_MODE_CANNY_WITH_COLOR;
                new_window_name_exists = true;
                break; 
            default:
                if(key_pressed>=0)
                    {
                        std::cout<<"Unrecognised key. Press h for help.\n";
                    }
                break;

        }

        if (new_window_name_exists)
        {    
            cv::setWindowTitle( old_window_name,new_window_name );
            new_window_name_exists = false;
        }

        
        
        
    }
    
    #ifdef BUILD_NV_PROFILER // If Nvidia Profiler is enabled for compile
        cudaProfilerStop();
    #endif

    cv::destroyWindow (old_window_name);
    d_img_color.release();
    d_img_gray.release();
    d_img_gray_blurred.release();
    d_edges.release();
    avgframe.release();
    shared_frame.release();
    canny.release();
    
    //printf("Control endline \n");
    return 0;
    
}

void display_help()
{
    std::cout<< "Application control keys:\n";
    std::cout<< "   Display Help: 'h'\n\n";
    std::cout<< "   No Filter              : 'a'\n";
    std::cout<< "   Gray                   : 's'\n";
    std::cout<< "   Gray w/ Gaussian       : 'd'\n";
    std::cout<< "   Canny Edge Detection   : 'f'\n";
    std::cout<< "   Canny w/ Colors        : 'g'\n\n";
    std::cout<< "   Blur sigma       : Increase: '+'\n"<<
                "                      Decrease: '-'\n\n"<<
                "   Blur window size : Increase: '4'\n"<<
                "                      Decrease: '1'\n\n"<<
                "   Edge Detection   : Lower threshold: Increase: '5'\n"<<
                "                                       Decrease: '2'\n\n"<<
                "                    : Upper threshold: Increase: '6'\n"<<
                "                                       Decrease: '3'\n\n"<<
                "   Exit application : 'Esc'\n";
}
