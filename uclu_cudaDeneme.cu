/* Based on the NVIDIA example
 * Yekta & Dogukan
 */
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <opencv2/opencv.hpp>

namespace cg = cooperative_groups;

/*
#include <helper_string.h>
#include <npp.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
*/

// Texture reference for reading image
/*
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

*/
#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

uchar4 char3_to_char4(unsigned char x,unsigned char y,unsigned char z)
{
    uchar4 result;
    result.x = (unsigned char)x;
    result.y = (unsigned char)y;
    result.z = (unsigned char)z;
    result.w = (unsigned char)0;
    return result;
}

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));

    if (Sum < 0)
    {
        return 0;
    }
    else if (Sum > 0xff)
    {
        return 0xff;
    }

    return (unsigned char) Sum;
}


__global__ void
SobelTex(unsigned char *pSobelOriginal, unsigned int Pitch,
         int w, int h, float fScale, unsigned char* array)
{
    //unsigned char *pSobel =
    //    (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    //for (int i = threadIdx.x; i < w; i += blockDim.x)
    //for (int i = 0; i < threadIdx.x*blockDim.x; i++)
    
    float r_weight = 0.2989f;
    float g_weight = 0.5870f;
    float b_weight = 0.1140f;
    
    
    //for (int j=0;j<3;j++)
    {
    //int i = (threadIdx.x+blockDim.x*blockIdx.x);
    /*
        unsigned char pix00 = tex2D(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D(tex, (float) i+1, (float) blockIdx.x+1);
        
     */
      int myRow = blockIdx.x/2; // exclude 0 & 961
      int myCol = threadIdx.x + (blockIdx.x-2*(blockIdx.x/2))*480; // exclude 0 & 961
      int i     = myCol+myRow*960;
      
      //i=threadIdx.x+blockIdx.x*blockDim.x;
                                 
      if (myRow>0&& myRow <959 && myCol>0 && myCol<959)   
      {
      unsigned char pix00 = (b_weight*array[3*(i-w-1)]  +g_weight*array[3*(i-w-1)+1]    +r_weight*array[3*(i-w-1)+2]);
      unsigned char pix01 = (b_weight*array[3*(i-1)  ]  +g_weight*array[3*(i-1)  +1]    +r_weight*array[3*(i-1)  +2]);
      unsigned char pix02 = (b_weight*array[3*(i+w-1)]  +g_weight*array[3*(i+w-1)+1]    +r_weight*array[3*(i+w-1)+2]);
      unsigned char pix10 = (b_weight*array[3*(i-w)  ]  +g_weight*array[3*(i-w)  +1]    +r_weight*array[3*(i-w)  +2]);
      unsigned char pix11 = (b_weight*array[3*(i)    ]  +g_weight*array[3*(i)    +1]    +r_weight*array[3*(i)    +2]);
      unsigned char pix12 = (b_weight*array[3*(i+w)  ]  +g_weight*array[3*(i+w)  +1]    +r_weight*array[3*(i+w)  +2]);
      unsigned char pix20 = (b_weight*array[3*(i-w+1)]  +g_weight*array[3*(i-w+1)+1]    +r_weight*array[3*(i-w+1)+2]);
      unsigned char pix21 = (b_weight*array[3*(i+1)  ]  +g_weight*array[3*(i+1)  +1]    +r_weight*array[3*(i+1)  +2]);
      unsigned char pix22 = (b_weight*array[3*(i+w+1)]  +g_weight*array[3*(i+w+1)+1]    +r_weight*array[3*(i+w+1)+2]);
      pSobelOriginal[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
      }
      /*else
      {
        pSobelOriginal[i] = (b_weight*array[3*(i)    ]  +g_weight*array[3*(i)    +1]    +r_weight*array[3*(i)    +2]);
      }*/
       
      //pSobelOriginal[i] =pix11;
      //pSobelOriginal[i+1] =0;
      //pSobelOriginal[i+2] =0;
    }
}
/*
// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(cv::Mat& cur_f, int iw, int ih, float fScale)
{

    uchar4* h_cur_f;
    
    h_cur_f = new uchar4[cur_f.cols*cur_f.rows];
    
    for(int i=0;i<cur_f.cols*cur_f.rows;i++)
    {
        h_cur_f[i]= char3_to_char4(cur_f.data[i*3+2],cur_f.data[i*3+1],cur_f.data[i*3]);
    }
    
    unsigned char* d_cur_f;
    
    cudaMalloc( (void**) &d_cur_f, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char));

    cudaChannelFormatDesc uchar4tex= cudaCreateChannelDesc<uchar4>();

    cudaMallocArray(&array, &uchar4tex, iw, ih);
    
    cudaMemcpyToArray(array, 0, 0, h_cur_f, 3*sizeof(unsigned char)*iw*ih, cudaMemcpyHostToDevice);
    
    cudaBindTextureToArray(tex, array);
    
    cudaMemcpy(d_cur_f,cur_f.data,3*cur_f.rows*cur_f.cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
    
    SobelTex<<<32, 32>>>(d_cur_f, iw, iw, ih, fScale);
    
    cudaDeviceSynchronize();

    cudaUnbindTexture(tex);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(cur_f.data, d_cur_f, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char), cudaMemcpyDeviceToHost); 
    
    cudaFree(d_cur_f);
    cudaFreeArray(array);

    cudaDeviceSynchronize();
    
    free(h_cur_f);
}
*/

__global__ void cuda_gray(cv::cuda::GpuMat gray_out, cv::cuda::GpuMat color_in)
{
    
    
    float r_weight = 0.2989f;
    float g_weight = 0.5870f;
    float b_weight = 0.1140f;
    
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    
    int idx_color = 3*(i% color_in.cols) + i/color_in.cols * color_in.step;
    //int idx_gray  =    i% gray_out.cols  + i/gray_out.cols ;
    
    
    gray_out.data[i] = (unsigned char)   (
                                                    b_weight*color_in.data[idx_color    ] +
                                                    g_weight*color_in.data[idx_color + 1] +
                                                    r_weight*color_in.data[idx_color + 2]
                                                );

    
}

void gray(cv::cuda::GpuMat& color_in, cv::cuda::GpuMat& gray_out)
{
    //unsigned char* d_cur_f;
    //unsigned char* d_in;
    
    //cudaMalloc( (void**) &d_cur_f, cur_f.rows*cur_f.cols*sizeof(unsigned char));
    //cudaMalloc( (void**) &d_in, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char));
    
    //cudaMemcpy(d_in, cur_f.data, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    //cv::cuda::GpuMat d_in(cur_f.rows,cur_f.cols,CV_8UC3);;
    //d_in.upload(cur_f);
    //cv::cuda::GpuMat d_in(cur_f);
    
    //std::cout<<"Control point 1" <<std::endl; 
    
    
    
    cuda_gray<<<(gray_out.rows*gray_out.cols)/32, 32>>>(gray_out, color_in);
    
    cudaDeviceSynchronize();
    //std::cout<<"color_in size: "<<color_in.cols<<"x"<<color_in.rows<<std::endl;
    //std::cout<<"gray_out size: "<<gray_out.cols<<"x"<<gray_out.rows<<std::endl;
    //std::cout<<"color_in step: "<<color_in.step<<"\ngray_out step : "<<gray_out.step<<std::endl;
    //std::cout<<"Control point 2" <<std::endl; 
    
    //cudaMemcpy(out_gray.data, d_cur_f, cur_f.rows*cur_f.cols*sizeof(unsigned char), cudaMemcpyDeviceToHost); 
    
    //cudaFree(d_cur_f);
    //cudaFree(d_in);

    //cudaDeviceSynchronize();

}

extern "C" void sobelFilter(cv::Mat& cur_f, int iw, int ih, float fScale, cv::Mat& out_gray)
{

    /*
    unsigned char* h_cur_f;
    
    h_cur_f = new unsigned char[cur_f.cols*cur_f.rows];
    
    for(int i=0;i<cur_f.cols*cur_f.rows;i++)
    {
        h_cur_f[i]= char3_to_char4(cur_f.data[i*3+2],cur_f.data[i*3+1],cur_f.data[i*3]);
    }
    */
    
    unsigned char* d_cur_f;
    unsigned char* d_in;
    //cv::Mat out_gray(cur_f.rows,cur_f.cols,CV_8UC1);
    
    cudaMalloc( (void**) &d_cur_f, cur_f.rows*cur_f.cols*sizeof(unsigned char));
    cudaMalloc( (void**) &d_in, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char));
    
    cudaMemcpy(d_in, cur_f.data, 3*cur_f.rows*cur_f.cols*sizeof(unsigned char), cudaMemcpyHostToDevice); 

    //cudaChannelFormatDesc uchar4tex= cudaCreateChannelDesc<unsigned char>();

    
    //cudaMallocArray(&array, &uchar4tex, iw, ih);
    
    //cudaMemcpyToArray(array, 0, 0, h_cur_f, sizeof(unsigned char)*iw*ih, cudaMemcpyHostToDevice);
    
    //cudaBindTextureToArray(tex, array);
    
    //cudaMemcpy(d_cur_f,h_cur_f, cur_f.rows*cur_f.cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
    
    dim3 Threads(16, 16);  // 1024 threads per block
    dim3 Blocks(60,60);    // 100 Blocks
    
    SobelTex<<<1920, 480>>>(d_cur_f, iw, iw, ih, fScale, d_in);
    
    cudaDeviceSynchronize();

    //cudaUnbindTexture(tex);
    
    //cudaDeviceSynchronize();
    
    cudaMemcpy(out_gray.data, d_cur_f, cur_f.rows*cur_f.cols*sizeof(unsigned char), cudaMemcpyDeviceToHost); 
    
    /*
    for(int i=0;i<cur_f.cols*cur_f.rows;i=i+3)
    {
        cur_f.data[i]=char4_to_char3(h_cur_f[i],);
        cur_f.data[i+1]=char4_to_char3(h_cur_f[i]);
        cur_f.data[i+2]=char4_to_char3(h_cur_f[i]);
    }
    
    */
    cudaFree(d_cur_f);
    cudaFree(d_in);
    //cudaFreeArray(array);

    cudaDeviceSynchronize();
    //return out_gray;
    //free(h_cur_f);
}


__global__ void cuda_combine(cv::cuda::GpuMat gray_in, cv::cuda::GpuMat color_in_out)
//cuda_gray(unsigned char *pSobelOriginal, unsigned char* array)
{
    
    int i= threadIdx.x+blockIdx.x*blockDim.x;
    
    int idx_color = 3*(i% color_in_out.cols) + i/color_in_out.cols * color_in_out.step;
    int idx_gray  =    i% gray_in.cols       + i/gray_in.cols      * gray_in.step;
    
    unsigned char intensity = gray_in.data[idx_gray];
    if (intensity>1.0f)
    {
        /*color_in_out.data[idx_color    ] = intensity;
        color_in_out.data[idx_color + 1] = intensity;
        color_in_out.data[idx_color + 2] = intensity;*/
        color_in_out.data[idx_color    ] = 255 - (short)color_in_out.data[idx_color    ];
        color_in_out.data[idx_color + 1] = 255 - (short)color_in_out.data[idx_color +1 ];
        color_in_out.data[idx_color + 2] = 255 - (short)color_in_out.data[idx_color +2 ];
        
    }
    //gray_out.data[idx_out] = (b_weight*color_in.data[3*(idx_in)    ]  +g_weight*color_in.data[3*(idx_in)    +1]    +r_weight*color_in.data[3*(idx_in)    +2]);

    
}

void combine(cv::cuda::GpuMat& color_in_out, cv::cuda::GpuMat& gray_in)
{
    cuda_combine<<<(color_in_out.rows*color_in_out.cols)/32, 32>>>(gray_in, color_in_out);
    
    cudaDeviceSynchronize();
}
