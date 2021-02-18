#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include "const.h"

extern "C"
{
    __global__ void adam(float* w, float* dw, float* s, float* d, float rate, float alpha, float beta, int iteration, int n)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < n)
        {
            s[tid] = alpha * s[tid] + (1 - alpha) * dw[tid];
            d[tid] = beta * d[tid] + (1 - beta) * dw[tid] * dw[tid];

            float _s = s[tid] / (1 - powf(alpha, iteration));
            float _d = d[tid] / (1 - powf(beta, iteration));
            w[tid] -= rate / (sqrtf(_d) + float_min) * _s;
            dw[tid] = 0;

            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void adaDelta(float* weights, float* dw, float* esq, float rate, float gamma, int n)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < n)
        {
            esq[tid] = gamma * esq[tid] + (1 - gamma) * powf(dw[tid], 2);
            weights[tid] -= rate / sqrtf(esq[tid] + 1e-54) * dw[tid];
            dw[tid] = 0;

            tid += blockDim.x * gridDim.x;
        }
    }


    __global__ void adaGrad(float* w, float* dw, float* history, float rate, int n)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < n)
        {
            history[tid] += powf(dw[tid], 2);
            w[tid] -= rate / sqrtf(history[tid] + 1e-54) * dw[tid];
            dw[tid] = 0;

            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ void gradientDescent(float* w, float* dw, float rate, int n)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < n)
        {
            w[tid] -= rate * dw[tid];
            dw[tid] = 0;

            tid += blockDim.x * gridDim.x;
        }
    }

}
