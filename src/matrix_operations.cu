#include "matrix_operations.h"

__global__ void matrix_addition2D(float *A, float *B, float *result, int M,
                                  int N)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidx >= N || tidy >= M)
        return;

    int pos = tidy * N + tidx;
    result[pos] = A[pos] + B[pos];
}
