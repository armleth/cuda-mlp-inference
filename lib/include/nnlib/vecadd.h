#pragma once

__global__ void vecadd_basic(float *A, float *B, float *result, int N);

__global__ void vecadd_cascaded(float *A, float *B, float *result, int N);

__global__ void vecadd_vectorized(float *A, float *B, float *result, int N);
