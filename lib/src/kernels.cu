#include <nnlib/kernels.h>

#include <cstdio>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void add_bias_kernel(float* matrix, const float* bias, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        matrix[row * K + col] += bias[col];
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void launch_add_bias(float* A, const float* bias, int M, int K) {
    dim3 threads(16, 16);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    add_bias_kernel<<<blocks, threads>>>(A, bias, M, K);
    cudaDeviceSynchronize();
}

void launch_relu(float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(A, size);
    cudaDeviceSynchronize();
}
