#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_descriptor.h"
#include "math.h"

#define sigmoid(x) (1 / (1 + expf(-x)))

#define tanh(x) (expf(x) - expf(x)) / (expf(x) + expf(x))

extern "C"
{
    __global__ void relu_forward(float* x, float* result, desc xDesc)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < xDesc.size)
        {
            result[i] = x[i] > 0 ? x[i] : 0;
            i += blockDim.x * gridDim.x;
        }

    }

    __global__ void relu_backward(float* x, float* dy, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            result[tid] = (x[tid] > 0 ? 1 : 0) * dy[tid];
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void sigmoid_forward(float* x, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            result[tid] = sigmoid(x[tid]);
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void sigmoid_backward(float* x, float* dy, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            result[tid] = sigmoid(x[tid]) * (1 - sigmoid(x[tid])) * dy[tid];
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void tanh_forward(float* x, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            result[tid] = tanh(x[tid]);
            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void tanh_backward(float* x, float* dy, float* result, desc xDesc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < xDesc.size)
        {
            result[tid] = (1 - powf(tanh(x[tid]), 2)) * dy[tid];
            tid += blockDim.x * gridDim.x;
        }
    }

}
