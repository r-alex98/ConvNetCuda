#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"

#define desc tensor_descriptor

extern "C"
{
    __global__ void transpose2d(float* x, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        int n = xDesc.height;
        int m = xDesc.width;
        while (tid < n * m)
        {
            int i = tid / n;
            int j = tid % n;
            result[tid] = x[j * m + i];

            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void rotate180(float* x, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            int sectorSize = xDesc.height * xDesc.width;
            int localI = tid % sectorSize;
            int sectorI = tid / (xDesc.batch * xDesc.channels);
            result[tid] = x[sectorI * sectorSize + sectorSize - localI - 1];
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void fill(float* x, float value, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            x[tid] = value;
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void sum(float* a, float* b, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            a[tid] += b[tid];
            tid += blockDim.x * gridDim.x;
        }
    }
    
    __global__ void findMax(float* x, float* max, desc xDesc)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        extern __shared__ float shData[];

        shData[tid] = x[i];

        __syncthreads();

        for (int s = 1; s < blockDim.x; s *= 2)
        {
            int index = 2 * s * tid;
            if (index < blockDim.x)
            {
                if (shData[index] < shData[index + s])
                {
                    shData[index] = shData[index + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            max[blockIdx.x * 2] = shData[0];
        }

        if (shData[0] == x[i])
        {
            max[blockIdx.x * 2 + 1] = tid;
        }

    }
}
