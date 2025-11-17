#pragma once

// Compute matrix addition
// A, B and result are MxN matrixes.
__global__ void matrix_addition2D(float *A, float *B, float *result, int M,
                                  int N);

// Compute matrix multiplication: result = A × B
// A is an M×N matrix, B is an N×K matrix, and result is an M×K matrix.
__global__ void matrix_multiplication(float *A, float *B, float *result, int M,
                                      int N, int K);
