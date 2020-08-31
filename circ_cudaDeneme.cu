
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

__global__ void inRange_kernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<unsigned char> dst,int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2)
{
  float y = blockIdx.x * blockDim.x + threadIdx.x;
  float x = blockIdx.y * blockDim.x + threadIdx.y;

  //if (x >= src.cols || y >= src.rows) return;

  uchar3 v = src(y, x);
  if (v.x >= lbc0 && v.x <= ubc0 && v.y >= lbc1 && v.y <= ubc1 && v.z >= lbc2 && v.z <= ubc2)
    dst(y,x) = (unsigned char)255;
  else
    dst(y,x) = (unsigned char)0;
}

void inRange_gpu(cv::cuda::GpuMat &src, cv::Scalar &lowerb, cv::Scalar &upperb,cv::cuda::GpuMat &dst)
{
  const int m = 32;
  int numRows = src.rows, numCols = src.cols;
  
  //printf("in row %d, in col %d, out row %d, out col %d \n",src.rows ,src.cols ,dst.rows ,dst.cols);
  
  //printf("Rows %d and cols %d \n",numRows,numCols);
  
  if (numRows == 0 || numCols == 0) return;
  // Attention! Cols Vs. Rows are reversed
  const dim3 gridSize(ceil((float)numRows / m), ceil((float)numCols / m), 1);
  const dim3 blockSize(m, m, 1);


  inRange_kernel<<<gridSize, blockSize>>>(src, dst, lowerb[0], upperb[0], lowerb[1], upperb[1],
                                          lowerb[2], upperb[2]);
  cudaDeviceSynchronize();
}


