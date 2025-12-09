#include <nnlib/matmul.h>

__global__ void matmul_basic(float *A, float *B, float *result, int M, int N, int K)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidx >= K || tidy >= M)
        return;

    float sum = 0;
    for (int i = 0; i < N; ++i)
        sum += A[tidy * N + i] * B[i * K + tidx];

    result[tidy * K + tidx] = sum;
}

__global__ void matmul_tiled(float *A, float *B, float *result, int M, int N, int K)
{
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++i)
    {
        int elt_position_row = i * TILE_WIDTH + threadIdx.x;
        if (y < M && elt_position_row < N)
            sh_A[threadIdx.y][threadIdx.x] = A[N * y + elt_position_row];
        else
            sh_A[threadIdx.y][threadIdx.x] = 0.0f;

        int elt_position_col = i * TILE_WIDTH + threadIdx.y;
        if (elt_position_col < N && x < K)
            sh_B[threadIdx.y][threadIdx.x] = B[elt_position_col * K + x];
        else
            sh_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
            sum += sh_A[threadIdx.y][j] * sh_B[j][threadIdx.x];

        __syncthreads();
    }

    if (x < K && y < M)
        result[y * K + x] = sum;
}
