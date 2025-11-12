#pragma once

// A and B are of shape MxN
__global__ void matrix_addition2D(float *A, float *B, float *result, int M,
                                  int N);

// Multiply A by B. A is of shape MxN and B is of shape NxK
// __global__ void matrix_multiplication(float *A, float *B, float *result, int
// M,
//                                       int N, int K)
// {
//
// }
