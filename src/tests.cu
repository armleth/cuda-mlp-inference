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

__host__ int tests_matrix_addition()
{
    int M = 2;
    int N = 2;
    int SIZE = M * N;

    float *A, *B, *result;
    cudaMallocManaged(&A, SIZE * sizeof(float));
    cudaMallocManaged(&B, SIZE * sizeof(float));
    cudaMallocManaged(&result, SIZE * sizeof(float));

    std::iota(A, A + SIZE, 1);
    std::iota(B, B + SIZE, 1);

    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y);

    matrix_addition2D<<<blocks_per_grid, threads_per_block>>>(A, B, result, M,
                                                              N);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool error = false;
    for (auto i = 0; i < SIZE; i++)
    {
        if ((error = A[i] + B[i] != result[i]))
        {
            std::cout << "ERROR at index " << i << ".\n";
            break;
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(result);
    cudaCheckError();

    if (!error)
    {
        std::cout << "Test completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "WARNING there were some errors.\n";
        return 1;
    }
}
