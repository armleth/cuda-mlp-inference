#include <cstdio>
#include <nnlib/kernels.h>

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

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void add_bias_kernel(float *matrix, const float *bias, int M, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K)
    {
        matrix[row * K + col] += bias[col];
    }
}

__global__ void relu_kernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void launch_matmul(const float *A, const float *B, float *C, int M, int N,
                   int K)
{
    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void launch_add_bias(float *A, const float *bias, int M, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    add_bias_kernel<<<blocks, threads>>>(A, bias, M, K);
    cudaDeviceSynchronize();
}

void launch_relu(float *A, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(A, size);
    cudaDeviceSynchronize();
}
