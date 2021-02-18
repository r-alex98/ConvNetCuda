#include "cuda_runtime.h"
#include "tensor_descriptor.h"

__forceinline__ __device__ int calcIndex(int b, int c, int i, int j, desc d)
{
	return b * d.channels * d.width * d.height + c * d.height * d.width + i * d.width + j;
}

__forceinline__ __device__ int calcIndex(int c, int i, int j, desc d)
{
	return c * d.height * d.width + i * d.width + j;
}

__forceinline__ __device__ int calcIndex(int i, int j, desc d)
{
	return i * d.width + j;
}
