/* Based on the NVIDIA example
 * Yekta & Dogukan
 */
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
 
 
__global__ void matrixMulCUDA(float* d_C, float* d_A, float* d_B, int wA, int wB)
{

    int hB=wA;
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int c_row = tx + bx*32;
    int c_col = ty + by*32;
    
    int a_row = c_row;
    int b_col = c_col;
    
    int a_begin = (a_row*wA);
    int a_end = (a_row+1)*wA;
    
    int b_begin = b_col;
    int b_end = b_col+(wB*(hB-1));
    
    int c_idx = ty + wB*tx + 32*(by + wB*bx); 
    
    int sum=0;
    
    for(int k=0;k<wA;k++)
    {
        sum+= d_A[a_begin+k]*d_B[b_begin+(wB*k)];    
    }
    
    d_C[c_idx]=sum;

    
}
 
 
 
 
 
void matrixMultiply(float* h_A,float* h_B, float* h_C, dim3 dimsA, dim3 dimsB)
{

    int mem_size_A,mem_size_B,mem_size_C;
    
    mem_size_A = (dimsA.x)*(dimsA.y)*sizeof(float);
    mem_size_B = (dimsB.x)*(dimsB.y)*sizeof(float);
    mem_size_C = (dimsA.x)*(dimsB.y)*sizeof(float);
    
    cudaError_t error;
    
    float *d_A, *d_B, *d_C;
    
    error = cudaMalloc((void **) &d_A, mem_size_A);
    error = cudaMalloc((void **) &d_B, mem_size_B);
    error = cudaMalloc((void **) &d_C, mem_size_C);
    
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    
    
    int block_size=32;
    
    dim3 threads(block_size, block_size);
    dim3 grid( ceil((float)dimsA.x / (float)threads.x), ceil((float)dimsB.y / (float)threads.y) );
    
    std::cout <<"Threads: " << threads.x << " " << threads.y <<" " << threads.z << std::endl;
    std::cout <<"Grids:   "<< grid.x << " " << grid.y <<" " << grid.z << std::endl;
    
        
    matrixMulCUDA <<< grid, threads >>> (d_C, d_A, d_B, dimsA.y, dimsB.y);
    
    cudaEvent_t start;
    error = cudaEventCreate(&start);
    
    cudaEvent_t stop;
    error = cudaEventCreate(&stop);
    
    error = cudaEventRecord(start, NULL);
    
    matrixMulCUDA <<< grid, threads >>> (d_C, d_A, d_B, dimsA.y, dimsB.y);

    
    cudaDeviceSynchronize();
   
    error = cudaEventRecord(stop, NULL);
    
    error = cudaEventSynchronize(stop);
    
    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
   
    printf("Total time for GPU %.3fms \n",msecTotal);
   
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    
}
 
 

/**
 * Program main 
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    
    printf("Please give the dimensions of the first matrix \n");
    
    int aX,aY,bX,bY;
    
    std::cin >> aX >>aY;
    
    printf("Please give the dimensions of the second matrix \n");
    
    std::cin >> bX >>bY;
    
    

    
    dim3 dimsA(aX,aY,1);
    
    dim3 dimsB(bX,bY,1);
    
    float *h_A = (float *)malloc(aX*aY*sizeof(float));
    float *h_B = (float *)malloc(bX*bY*sizeof(float));
    float *h_C = (float *)malloc(aX*bY*sizeof(float));
    
    for(int i=0;i<(aX*aY);i++)
    {   
        h_A[i]=i+1;
    }
    
        for(int i=0;i<(bX*bY);i++)
    {
        h_B[i]=i+1;
    }

    matrixMultiply(h_A, h_B, h_C, dimsA, dimsB);
    
    /*
    for(int i=0;i<aX*bY;i++)
    {
        if((i%bY)==0)
            std::cout << std::endl;
        std::cout << h_C[i] << " ";        
    }
    */      
  
    free(h_A);
    free(h_B);
    free(h_C);

    //exit(matrix_result);

}
