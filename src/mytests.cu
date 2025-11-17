%
    % writefile cuda_example.cu
#include <iostream>
#include <numeric>

#define WORK_PER_THREAD 4

          __global__ void
          matrix_addition2D(float *A, float *B, float *result, int M, int N)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    if (tidx * WORK_PER_THREAD >= N || tidy >= M)
        return;

#pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i)
    {
        int pos = tidy * N + tidx * WORK_PER_THREAD + i;
        result[pos] = A[pos] + B[pos];
    }
}

__global__ void matrix_addition1D(float *A, float *B, float *result, int N)
{
    int threadIdxBase = blockDim.x * blockIdx.x + threadIdx.x;
    int start = threadIdxBase * WORK_PER_THREAD;

    if (start + WORK_PER_THREAD - 1 >= N)
        return;

#pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i)
    {
        int pos = start + i;
        result[pos] = A[pos] + B[pos];
    }
}

__global__ void matrix_addition1D_stripped(float *A, float *B, float *result,
                                           int N)
{
    int threadIdxBase = blockDim.x * blockIdx.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

#pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; ++i)
    {
        int pos = threadIdxBase + i * totalThreads;
        result[pos] = A[pos] + B[pos];
    }
}

__global__ void matrix_addition1D_basic(float *A, float *B, float *result,
                                        int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = A[i] + B[i];
}

__global__ void matrix_addition1D_basic_blocked(float *A, float *B,
                                                float *result, int N)
{
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    int j = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2 + 1;

    // printf("%d ", i);

    result[i] = A[i] + B[i];
    result[j] = A[j] + B[j];
}

__global__ void matrix_addition1D_basic_stripped(float *A, float *B,
                                                 float *result, int N)
{
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int j = blockIdx.x * blockDim.x * 2 + threadIdx.x + blockDim.x;

    result[i] = A[i] + B[i];
    result[j] = A[j] + B[j];
}

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
    int M = 1000;
    int N = 1000;
    int SIZE = M * N;

    float *A = (float *)malloc(SIZE * sizeof(float));
    float *B = (float *)malloc(SIZE * sizeof(float));
    float *result = (float *)malloc(SIZE * sizeof(float));

    std::iota(A, A + SIZE, 1);
    std::iota(B, B + SIZE, 1);

    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_result, SIZE * sizeof(float));

    cudaMemcpy(d_A, A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    /*
    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid(
        (N + WORK_PER_THREAD * threads_per_block.x - 1) / (WORK_PER_THREAD *
    threads_per_block.x), (M + threads_per_block.y - 1) / threads_per_block.y);

    std::cout << "threads_per_block: (" << threads_per_block.x << ", " <<
    threads_per_block.y << ")\n"; std::cout << "blocks_per_grid: (" <<
    blocks_per_grid.x << ", " << blocks_per_grid.y << ")\n";

    matrix_addition2D<<<blocks_per_grid, threads_per_block>>>(d_A, d_B,
                                                              d_result, M, N);
    */

    const int threads_per_block = 256;
    const int totalThreads = (SIZE + 1) / 2;
    const int blocks_per_grid =
        (totalThreads + threads_per_block - 1) / threads_per_block;

    std::cout << "threads_per_block: " << threads_per_block << "\n";
    std::cout << "blocks_per_grid: " << blocks_per_grid << "\n";

    matrix_addition1D_basic_blocked<<<blocks_per_grid, threads_per_block>>>(
        d_A, d_B, d_result, SIZE);

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
