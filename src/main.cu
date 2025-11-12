#include <cstdlib>
#include <iostream>
#include <numeric>

#include "./matrix_operations.h"

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

int main()
{
    int M = 2;
    int N = 2;
    int SIZE = M * N;

    float *A = (float *)malloc(SIZE * sizeof(float));
    float *B = (float *)malloc(SIZE * sizeof(float));
    float *result = (float *)malloc(SIZE * sizeof(float));

    std::iota(A, A + SIZE, 1);
    std::iota(B, B + SIZE, 1);

    float *d_A, d_B, d_result;
    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_result, SIZE * sizeof(float));

    cudaMemcpy(d_A, A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y);

    matrix_addition2D<<<blocks_per_grid, threads_per_block>>>(d_A, d_B,
                                                              d_result, M, N);
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(result, d_result, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    bool error = false;
    for (auto i = 0; i < SIZE; i++)
    {
        if ((error = A[i] + B[i] != result[i]))
        {
            std::cout << "ERROR at index " << i << ".\n";
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    cudaCheckError();

    free(A);
    free(B);
    free(result);

    if (!error)
    {
        printf("Test completed successfully.\n");
        return 0;
    }
    else
    {
        printf("WARNING there were some errors.\n");
        return 1;
    }
}
