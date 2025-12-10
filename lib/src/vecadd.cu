#include <cstdio>
#include <nnlib/vecadd.h>

__global__ void vecadd_basic(float *A, float *B, float *result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        result[idx] = A[idx] + B[idx];
}

__global__ void vecadd_cascaded(float *A, float *B, float *result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride)
        result[i] = A[i] + B[i];
}

__global__ void vecadd_vectorized(float *A, float *B, float *result, int N)
{
    float4 *A4 = reinterpret_cast<float4 *>(A);
    float4 *B4 = reinterpret_cast<float4 *>(B);
    float4 *result4 = reinterpret_cast<float4 *>(result);

    int N4 = N / 4;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N4; i += stride)
    {
        float4 a_val = A4[i];
        float4 b_val = B4[i];
        float4 res_val;

        res_val.x = a_val.x + b_val.x;
        res_val.y = a_val.y + b_val.y;
        res_val.z = a_val.z + b_val.z;
        res_val.w = a_val.w + b_val.w;

        result4[i] = res_val;
    }

    int remainder_start = N4 * 4 + idx;

    if (remainder_start < N)
    {
        for (int i = N4 * 4 + idx; i < N; i += stride)
            result[i] = A[i] + B[i];
    }
}
