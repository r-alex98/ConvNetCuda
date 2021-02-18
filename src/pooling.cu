#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"

#define desc tensor_descriptor

extern "C"
{
    __global__ void maxPool(float* x, float* result, float* maxIndexes, int poolSize, int stride, desc x_desc, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < res_desc.size)
        {
            int b = tid / (x_desc.channels * res_desc.height * res_desc.width);
            int c = tid % (x_desc.channels * res_desc.height * res_desc.width) / (res_desc.height * res_desc.width);
            int kernelLocalNum = tid % (res_desc.height * res_desc.width);

            int startI = kernelLocalNum / res_desc.width * stride;
            int startJ = kernelLocalNum % res_desc.width * stride;

            float max = 1e-54;
            int i = 0;
            int j = 0;
            for (int ki = startI; ki < startI + poolSize; ki++)
            {
                for (int kj = startJ; kj < startJ + poolSize; kj++)
                {
                    float element = x[c * x_desc.height * x_desc.width + ki * x_desc.width + kj + b * x_desc.channels * x_desc.height * x_desc.height];
                    if (element > max)
                    {
                        max = element;
                        i = ki;
                        j = kj;
                    }
                }
            }

            result[tid] = max;
            maxIndexes[tid] = b * x_desc.channels * x_desc.height * x_desc.width + c * x_desc.height * x_desc.width + x_desc.width * i + j;

            tid += blockDim.x * gridDim.x;
        }

    }

    __global__ void maxPoolDx(float* dy, float* maxIndexes, float* dx, desc dxDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < dxDesc.size)
        {
            dx[(int)maxIndexes[tid]] = dy[tid];

            tid += blockDim.x * gridDim.x;
        }
    }

}
