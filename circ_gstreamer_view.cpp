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

#define DISPLAY_MODE_NO_FILTER 0
#define DISPLAY_MODE_RGB_BLUR 1
#define DISPLAY_MODE_BINARY_BALL 2
#define DISPLAY_MODE_BALL_WITH_COLOR 3

std::string get_tegra_pipeline(int width, int height, int fps) {

return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458,format=(string)I420, framerate=(fraction)"+std::to_string(fps)+"/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)"+std::to_string(width)+", height=(int)"+std::to_string(height)+", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink ";
}

void inRange_gpu(cv::cuda::GpuMat &src, cv::Scalar &lowerb, cv::Scalar &upperb,cv::cuda::GpuMat &dst);

void display_help();

int main(int argc, char** argv) {

    int display_mode = DISPLAY_MODE_NO_FILTER;
    int key_pressed=-2;
    
    int l1=0;
    int l2=70;
    int l3=0;
    int h1=40;
    int h2=165;
    int h3=100;
    cv::Point anchor(-1, -1);
    cv::Mat ker1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::Mat ker2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));

    
    int blur_size = 5;
    float blur_sigma = 1.7f;
    
    float small_edge_thresh = 60.0f;
    float large_edge_thresh = 80.0f;

    
    
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end   = std::chrono::high_resolution_clock::now();
    auto t_elaps = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start);
    float cur_fps = 0;
    int count = 0;
    std::string new_window_name;
    std::string display_mode_string;

    
    
    int WIDTH = 1280;
    int HEIGHT = 720;
    
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, 30);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    
    cv::cuda::HostMem shared_frame(HEIGHT, WIDTH, CV_8UC3, cv::cuda::HostMem::SHARED);
    cv::Mat avgframe = shared_frame.createMatHeader();
    cv::cuda::GpuMat gpuMat = shared_frame.createGpuMatHeader();
    
    
    std::string old_window_name="Display window";
    cv::namedWindow(old_window_name, cv::WINDOW_OPENGL);
    cv::resizeWindow(old_window_name, WIDTH, HEIGHT);
       
    float dp = 15;
    float minDist = 100; //If it is small it causes non circular detections.
    int cannyThreshold = 200;//Not a big deal
    int accThreshold = 20; //Important to find circless
    int minRadius = 10; // kind of important
    int maxRadius = WIDTH/4;//std::max(WIDTH, HEIGHT);

    std::vector<cv::Vec3f> gpuCircles;
    cv::cuda::GpuMat gpuResult;
    
    cv::Mat greyMat(WIDTH, HEIGHT, CV_8UC1);
    //gpuMat.upload(img);
    
    int thickness =2;
    int control=0;
    
    cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE , CV_8UC1, ker2, anchor, 2);
    cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker1, anchor, 2);
    //cv::Ptr<cv::cuda::Filter> close = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, ker2, anchor, 5);
    
    cv::Mat dummy(WIDTH, HEIGHT, CV_8UC1);
    
    cv::Ptr<cv::cuda::Filter> gaussian = cv::cuda::createGaussianFilter(CV_8UC3,CV_8UC3,cv::Size(blur_size,blur_size),blur_sigma,blur_sigma);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(small_edge_thresh, large_edge_thresh);
    cv::cuda::GpuMat blurred(HEIGHT,WIDTH,CV_8UC3);
    cv::cuda::GpuMat binary(HEIGHT,WIDTH,CV_8UC1);
    
    
    cv::cuda::GpuMat out(HEIGHT, WIDTH, CV_8UC1);

    
    for(;;)
    {
        //You can take the next three lines out of for loop, once the parameters are found, increases the frame rate from 6 to 8
        cv::Ptr<cv::cuda::HoughCirclesDetector> gpuDetector = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, accThreshold, minRadius, maxRadius);
        cv::Scalar lower(l1,l2,l3);
        cv::Scalar higher(h1,h2,h3);
        //Can be taken out of for loop
        
        
        cap >> avgframe;
    

        if(display_mode>=DISPLAY_MODE_NO_FILTER)
            gaussian->apply(gpuMat, blurred);

        //canny->detect(dum2, dum2);
        
        if(display_mode>=DISPLAY_MODE_RGB_BLUR)
            inRange_gpu(blurred,lower,higher,binary);


        
        //canny->detect(out, dum2);
        
        if(display_mode>=DISPLAY_MODE_BINARY_BALL)
        {
            erode->apply(binary, out);
            dilate->apply(out, out);
            
            //close->apply(out, out);
            
            //close->apply(out, out);
            //erode->apply(out, out);
        

            gpuDetector->detect(out, gpuResult);
            gpuCircles.resize(gpuResult.size().width);
            
            out.download(dummy);
            
            if (!gpuCircles.empty())
                {
                gpuResult.row(0).download(cv::Mat(gpuCircles).reshape(3, 1));
                control=0;
                }
            else
                {
                if (control == 0)
                    printf("No circles have been found dude \n");
                    control=1;
                }
                
                
            for( size_t i = 0; i < gpuCircles.size(); i++ )
            {
                     cv::Point center(cvRound(gpuCircles[i][0]), cvRound(gpuCircles[i][1]));
                     int radius = cvRound(gpuCircles[i][2]);
                     // draw the circle center
                     //circle( img, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
                     // draw the circle outline
                     circle( avgframe, center, radius, cv::Scalar(150,150,150), thickness, 8, 0 );
            }
        }
        
        t_end   = std::chrono::high_resolution_clock::now();
        t_elaps = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start);
        if(t_elaps.count()>=1000000.0f)
        {    
            // Calculate Average FPS over the last second
            t_start = std::chrono::high_resolution_clock::now();
            cur_fps = 1000000.0f/(t_elaps.count()/(float)count);
            // Frame counter between
            count = 0;
        }
        else
        {
            count++;
        }

        
        switch(display_mode)
        {
            case DISPLAY_MODE_NO_FILTER:
                cv::imshow(old_window_name, gpuMat);
                display_mode_string = "No Filter                |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            case DISPLAY_MODE_RGB_BLUR:
                cv::imshow(old_window_name, blurred);
                display_mode_string = "Blurred Image               |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            case DISPLAY_MODE_BINARY_BALL:
                cv::imshow(old_window_name, binary);
                display_mode_string = "Binary Ball Image       |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            case DISPLAY_MODE_BALL_WITH_COLOR:
                cv::imshow(old_window_name, gpuMat);
                display_mode_string = "Ball detection          |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
            default:
                cv::imshow(old_window_name, gpuMat);
                display_mode_string = "No Filter                |";
                new_window_name = display_mode_string + " FPS: "+std::to_string(cur_fps);
                break;
        }
        

        key_pressed=cv::waitKey(1);
        
        cv::setWindowTitle( old_window_name,new_window_name );

        // Keys to find color threshold  Q-W for l1, A-S for l2, X-C for l3, 7-8 for h1, 4-5 for h2, 1-2 for h3
        // Keys to find color threshold  Q-W for l1, A-S for l2, X-C for l3, 7-8 for h1, 4-5 for h2, 1-2 for h3
        // Keys to find color threshold  Q-W for l1, A-S for l2, X-C for l3, 7-8 for h1, 4-5 for h2, 1-2 for h3
        /*
                switch(key_pressed)
        {
            case 113:
                l1=l1+5;
                printf("l1 is %d \n",l1);
                break;
            case 119: 
                l1=l1-5;
                printf("l1 is %d \n",l1);
                break;
                
            case 97:
                l2=l2+5;
                printf("l2 is %d \n",l2);
                break;
            case 115: 
                l2=l2-5;
                printf("l2 is %d \n",l2);
                break;
                
            case 120:
                l3=l3+5;
                printf("l3 is %d \n",l3);
                break;
            case 99: 
                l3=l3-5;
                printf("l3 is %d \n",l3);
                break;
                
            case 55:
                h1=h1+5;
                printf("h1 is %d \n",h1);
                break;
            case 56: 
                h1=h1-5;
                printf("h1 is %d \n",h1);
                break;
                
            case 52:
                h2=h2+5;
                printf("h2 is %d \n",h2);
                break;
            case 53: 
                h2=h2-5;
                printf("h2 is %d \n",h2);
                break;
            
            case 49:
                h3=h3+5;
                printf("h3 is %d \n",h3);
                break;
            case 50: 
                h3=h3-5;
                printf("h3 is %d \n",h3);
                break;
                
            case 104: // 'h'
                display_help();
                break;
                
            case 116:
                display_mode=DISPLAY_MODE_NO_FILTER;
                break;
            case 121:
                display_mode=DISPLAY_MODE_RGB_BLUR;
                break; 
            case 117:
                display_mode=DISPLAY_MODE_BINARY_BALL;
                break;  
            case 105:
                display_mode=DISPLAY_MODE_BALL_WITH_COLOR;
                break;  
  
            default:
                if(key_pressed>=0)
                    {
                        std::cout<<"Unrecognised key. Press h for help.\n";
                    }
                break;
        }
        */
        
        // Keys to find hough parameters  7-8 for dp, 4-5 for minDist, 1-2 for canny, Q-W for acc, A-S for minRad
        // Keys to find hough parameters  7-8 for dp, 4-5 for minDist, 1-2 for canny, Q-W for acc, A-S for minRad
        // Keys to find hough parameters  7-8 for dp, 4-5 for minDist, 1-2 for canny, Q-W for acc, A-S for minRad
        // Keys to find hough parameters  7-8 for dp, 4-5 for minDist, 1-2 for canny, Q-W for acc, A-S for minRad
        
        //acc and dp are the most important ones
        switch(key_pressed)
            {
                case 55:
                    dp++;
                    printf("dp is %f \n",dp);
                    break;
                case 56: 
                    dp--;
                    printf("dp is %f \n",dp);
                    break;
                    
                case 52:
                    minDist=minDist+5;
                    printf("minDist is %f \n",minDist);
                    break;
                case 53: 
                    minDist=minDist-5;
                    printf("minDist is %f \n",minDist);
                    break;
                    
                case 49:
                    cannyThreshold=cannyThreshold+5;
                    printf("cannyThreshold is %d \n",cannyThreshold);
                    break;
                case 50: 
                    cannyThreshold=cannyThreshold-5;
                    printf("cannyThreshold is %d \n",cannyThreshold);
                    break;
                    
                case 113:
                    accThreshold=accThreshold+5;
                    printf("accThreshold is %d \n",accThreshold);
                    break;
                case 119: 
                    accThreshold=accThreshold-5;
                    printf("accThreshold is %d \n",accThreshold);
                    break;
                
                case 104: // 'h'
                    display_help();
                    break;
                    
                case 97:
                    minRadius=minRadius+3;
                    printf("minRadius is %d \n",minRadius);
                    break;
                case 115: 
                    minRadius=minRadius-3;
                    printf("minRadius is %d \n",minRadius);
                    break;
           
                case 116:
                    display_mode=DISPLAY_MODE_NO_FILTER;
                    break;
                case 121:
                    display_mode=DISPLAY_MODE_RGB_BLUR;
                    break; 
                case 117:
                    display_mode=DISPLAY_MODE_BINARY_BALL;
                    break;  
                case 105:
                    display_mode=DISPLAY_MODE_BALL_WITH_COLOR;
                    break;    
     
      
                default:
                    if(key_pressed>=0)
                        {
                            std::cout<<"Unrecognised key. Press h for help.\n";
                        }
                    break;
            }
            
    }
    
    return 0;
}


void display_help()
{
    std::cout<< "   Display Help                        :'h'\n";
    std::cout<< "   No Filter                           : 't'\n";
    std::cout<< "   RGB Blurred                         : 'y'\n";
    std::cout<< "   Binary optimized for green          : 'u'\n";
    std::cout<< "   Detected ball with colourd input    : 'ı'\n";
   
}
