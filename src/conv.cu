#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"
#include "tensor_ind.cu"
#include <iostream>

extern "C"
{
    __global__ void im2Col(float* image, float* res, desc image_desc, desc res_desc, int convByRow, int kernelSize)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        int i = tid / res_desc.width;
        int j = tid % res_desc.width;

        int curC = i / (kernelSize * kernelSize);
        int curB = j / (convByRow * convByRow);

        int kernelNumber = j;
        int kernelStartPointI = kernelNumber % (convByRow * convByRow) / convByRow;
        int kernelStartPointJ = kernelNumber % (convByRow * convByRow) % convByRow;

        int kernelIndex = i % (kernelSize * kernelSize);
        int kernelI = kernelIndex / kernelSize;
        int kernelJ = kernelIndex % kernelSize;

        int h1 = kernelStartPointI + kernelI;
        int w1 = kernelStartPointJ + kernelJ;
        res[i * res_desc.width + j] = image[curB * image_desc.height * image_desc.width * image_desc.channels + curC * image_desc.height * image_desc.width + image_desc.width * h1 + w1];
    }

    __global__ void col2Im(float* x, float* result, desc x_desc, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < res_desc.size)
        {
            int i = tid / x_desc.width;
            int j = tid % x_desc.width;
            
            int wh = res_desc.width * res_desc.height;
            
            int b = j / wh;
            int c = i;
            int h = j % wh / wh;
            int w = j % wh % wh;
            
            int res_i = calcIndex(b, c, h, w, res_desc);
            int x_i = calcIndex(i, j, x_desc);

            result[res_i] = x[x_i];

            tid += blockDim.x * gridDim.x;
        }

    }

    __global__ void to2DByRows(float* x, float* result, desc x_desc, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < x_desc.size)
        {
            int i = tid / res_desc.width;
            int j = tid % res_desc.width;

            int c = i;
            int b = j / (x_desc.width * x_desc.width);
            int iNum = j % (x_desc.height * x_desc.width);
            int h = iNum / x_desc.height;
            int w = iNum % x_desc.height;

            result[tid] = x[c * x_desc.height * x_desc.width + h * x_desc.width + w + b * x_desc.channels * x_desc.height * x_desc.width];

            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void to2DByColumns(float* x, float* result, desc x_desc, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < x_desc.size)
        {
            int i = tid / res_desc.width;
            int j = tid % res_desc.width;

            int c = j;
            int b = i / (x_desc.height * x_desc.height);
            int iNum = i % (x_desc.height * x_desc.width);
            int h = iNum / x_desc.width;
            int w = iNum % x_desc.width;

            result[tid] = x[c * x_desc.height * x_desc.width + h * x_desc.width + w + b * x_desc.channels * x_desc.height * x_desc.width];

            tid += blockDim.x * gridDim.x;
        }

    }

    __global__ void reshapeForBatches(float* x, float* res, desc x_desc, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < x_desc.size)
        {
            int i = tid / x_desc.width;
            int j = tid % x_desc.width;

            int c = i;
            int b = j / (res_desc.width * res_desc.width);
            int iNum = j % (res_desc.height * res_desc.width);
            int h = iNum / res_desc.height;
            int w = iNum % res_desc.height;

            res[c * res_desc.height * res_desc.width + h * res_desc.width + w + b * res_desc.channels * res_desc.height * res_desc.width] = x[tid];

            tid += blockDim.x * gridDim.x;
        }
    }

}
