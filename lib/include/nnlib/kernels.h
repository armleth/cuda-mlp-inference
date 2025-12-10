#pragma once

__global__ void vecadd_basic(float *A, float *B, float *result, int N);

__global__ void vecadd_cascaded(float *A, float *B, float *result, int N);

__global__ void vecadd_vectorized(float *A, float *B, float *result, int N);

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);

void launch_add_bias(float* A, const float* bias, int M, int K);

void launch_relu(float* A, int size);
