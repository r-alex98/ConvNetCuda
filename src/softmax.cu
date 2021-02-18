#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"
#include <iostream>
#include <math.h>
#include "const.h"

extern "C"
{
    __global__ void softmax(float* x, float* max, float* y, desc xDesc)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        extern __shared__ float denominator[];

        float exp = expf(x[i] - max[bid * 2]);
        denominator[tid] = exp;

        __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2)
        {
            int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                denominator[index] += denominator[index + s];
            }
            __syncthreads();
        }

        y[i] = exp / (denominator[0] + float_min);
    }

    __global__ void softmaxDx(float* y, float* dy, float* dx, desc yDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < yDesc.size)
        {
            float sum = 0;
            int sizePerBatch = yDesc.height * yDesc.channels * yDesc.width;
            int b = tid / sizePerBatch;                                                
            int i = b * sizePerBatch + tid % sizePerBatch;                              
            for (int j = b * sizePerBatch; j < b * sizePerBatch + sizePerBatch; j++)
            {
                float d;
                if (i == j)
                {
                    d = y[j] * (1 - y[i]);
                }
                else d = -y[i] * y[j];
                sum += d * dy[j];
            }
            dx[i] = sum;
            tid += blockDim.x * gridDim.x;
        }

    }

}
