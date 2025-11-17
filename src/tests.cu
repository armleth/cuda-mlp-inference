#include <iostream>
#include <numeric>
#include <vector>

#include "./matrix_operations.h"
#include "./tests.h"

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
            std::cout << "tests_matrix_addition: ERROR at index " << i << ".\n";
            break;
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(result);
    cudaCheckError();

    if (!error)
    {
        std::cout << "tests_matrix_addition: completed successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "tests_matrix_addition: there were some errors.\n";
        return 1;
    }
}

__host__ int tests_matrix_multiplication_basic()
{
    // A = [ 1 2 3 ]
    //     [ 4 5 6 ]   (2x3)
    //
    // B = [  7  8 ]
    //     [  9 10 ]
    //     [ 11 12 ]   (3x2)
    //
    // result = A * B = (2x2)
    // result = [ 58  64 ]
    //     [139 154]

    int M = 2;
    int N = 3;
    int K = 2;

    int sizeA = M * N;
    int sizeB = N * K;
    int sizeResult = M * K;

    float *A, *B, *result;
    cudaMallocManaged(&A, sizeA * sizeof(float));
    cudaMallocManaged(&B, sizeB * sizeof(float));
    cudaMallocManaged(&result, sizeResult * sizeof(float));

    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 3.0f;
    A[3] = 4.0f;
    A[4] = 5.0f;
    A[5] = 6.0f;

    B[0] = 7.0f;
    B[1] = 8.0f;
    B[2] = 9.0f;
    B[3] = 10.0f;
    B[4] = 11.0f;
    B[5] = 12.0f;

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);

    matrix_multiplication<<<blocks_per_grid, threads_per_block>>>(A, B, result,
                                                                  M, N, K);
    cudaDeviceSynchronize();
    cudaCheckError();

    float expected[4] = { 58.0f, 64.0f, 139.0f, 154.0f };

    bool error = false;
    const float eps = 1e-5f;
    for (int i = 0; i < sizeResult; ++i)
    {
        if (fabs(result[i] - expected[i]) > eps)
        {
            std::cout << "tests_matrix_multiplication_basic: ERROR at index "
                      << i << ". Got " << result[i] << ", expected "
                      << expected[i] << ".\n";
            error = true;
            break;
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(result);
    cudaCheckError();

    if (!error)
    {
        std::cout << "tests_matrix_multiplication_basic: test completed "
                     "successfully.\n";
        return 0;
    }
    else
    {
        std::cout << "tests_matrix_multiplication_basic: WARNING there were "
                     "some errors.\n";
        return 1;
    }
}

__host__ int tests_matrix_multiplication_large()
{
    int M = 6;
    int N = 8;
    int K = 25;

    int sizeA = M * N;
    int sizeB = N * K;
    int sizeC = M * K;

    float *A, *B, *C;
    cudaMallocManaged(&A, sizeA * sizeof(float));
    cudaMallocManaged(&B, sizeB * sizeof(float));
    cudaMallocManaged(&C, sizeC * sizeof(float));

    std::iota(A, A + sizeA, 1.0f);
    std::iota(B, B + sizeB, 1.0f);

    std::vector<float> C_ref(sizeC, 0.0f);

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < K; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i)
                sum += A[row * N + i] * B[i * K + col];
            C_ref[row * K + col] = sum;
        }
    }

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);

    matrix_multiplication<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N,
                                                                  K);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool error = false;
    const float eps = 1e-4f;

    for (int i = 0; i < sizeC; ++i)
    {
        if (fabs(C[i] - C_ref[i]) > eps)
        {
            std::cout << "tests_matrix_multiplication_large: ERROR at index "
                      << i << ". GPU=" << C[i] << ", CPU=" << C_ref[i] << "\n";
            error = true;
            break;
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaCheckError();

    if (!error)
    {
        std::cout << "tests_matrix_multiplication_large: test passed.\n";
        return 0;
    }
    else
    {
        std::cout
            << "tests_matrix_multiplication_large: WARNING test failed.\n";
        return 1;
    }
}
