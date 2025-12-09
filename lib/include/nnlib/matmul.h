#pragma once

#define TILE_WIDTH 32

__global__ void matmul_basic(float *A, float *B, float *result, int M, int N, int K);

__global__ void matmul_tiled(float *A, float *B, float *result, int M, int N, int K);
