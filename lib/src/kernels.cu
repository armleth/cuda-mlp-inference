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

__global__ void convolution_2d_square_kernel(float *input, float *output,
                                             unsigned int width,
                                             unsigned int height, float *mask,
                                             unsigned int kernel_size,
                                             unsigned int stride,
                                             unsigned int padding)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;

    if (out_col < output_width && out_row < output_height)
    {
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                // Corresponding input coordinates
                int in_row = out_row * stride + ky - padding;
                int in_col = out_col * stride + kx - padding;

                if (in_row >= 0 && in_row < height && in_col >= 0
                    && in_col < width)
                    sum += input[in_row * width + in_col]
                        * mask[ky * kernel_size + kx];
            }
        }

        output[out_row * output_width + out_col] = sum;
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
