#include "cuda_runtime.h"
#include "tensor_descriptor.h"
#include "device_launch_parameters.h"

extern "C"
{
    __global__ void pad(float* x, float* result, int value, int w, int chw, int hw, desc res_desc)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        while (tid < res_desc.size)
        {
            int b = tid / chw;
            int c = tid % chw / hw;
            int localI = tid % chw % hw;
            int h = localI / w;
            int _w = localI % w;

            int i = c * res_desc.height * res_desc.width + (h + value) * res_desc.width + (_w + value) + b * res_desc.channels * res_desc.height * res_desc.width;
            result[i] = x[tid];
            tid += blockDim.x * gridDim.x;
        }

    }
}
