#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"
#include "const.h"
#include <iostream>
#include <math.h>

extern "C"
{
    __global__ void mean_squared_error(float* o, float* t, float* loss, desc d)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        extern __shared__ float shData[];

        float dif = o[i] - t[i];
        shData[tid] = dif * dif;

        __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2)
        {
            int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                shData[index] += shData[index + s];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            loss[bid] = shData[0] / (d.channels * d.height * d.width);
        }


    }

    __global__ void mean_squared_dy(float* o, float* t, float* dy, desc d)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < d.size)
        {
            dy[tid] = 2.f / (d.size / d.batch) * (o[tid] - t[tid]);
            tid += blockDim.x * gridDim.x;
        }

    }

    __global__ void cross_entropy(float* o, float* t, float* loss, desc d)
    {
        extern __shared__ float shData[];

        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        shData[tid] = t[i] * logf(o[i] + float_min);

        __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2)
        {
            int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                shData[index] += shData[index + s];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            loss[bid] = -shData[0];
        }

    }

    __global__ void cross_entropy_dy(float* o, float* t, float* dy, desc d)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < d.size)
        {
            dy[tid] = -t[tid] / (o[tid] + float_min);
            tid += blockDim.x * gridDim.x;
        }
    }

}
