#include "catch.hpp"
#include <nnlib/vecadd.h>
#include <nnlib/tensor.h>
#include <nnlib/matmul.h>
#include <nnlib/kernels.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "assert.h"

/* approx_equal: helper for floating point comparison */
bool approx_equal(float a, float b, float epsilon = 0.001f) {
    return std::abs(a - b) < epsilon;
}

TEST_CASE("Kernel: ReLU Activation (Tensor2D)", "[kernel][relu]") {
    auto t = std::make_shared<Tensor2D>(1, 5);
    std::vector<float> input = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    t->set_data(input);

    unsigned int threads = 256;
    unsigned int blocks = (t->size() + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(t->data(), t->size());
    cudaDeviceSynchronize();

    REQUIRE(t->data()[0] == 0.0f);
    REQUIRE(t->data()[1] == 0.0f);
    REQUIRE(t->data()[2] == 0.0f);
    REQUIRE(t->data()[3] == 1.0f);
    REQUIRE(t->data()[4] == 10.0f);
}

TEST_CASE("Kernel: Matrix Multiplication (Tensor2D)", "[kernel][matmul]") {
    // A (2x3)
    // 1 2 3
    // 4 5 6
    auto A = std::make_shared<Tensor2D>(2, 3);
    A->set_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    // B (3x2)
    // 7  8
    // 9  10
    // 11 12
    auto B = std::make_shared<Tensor2D>(3, 2);
    B->set_data({7.f, 8.f, 9.f, 10.f, 11.f, 12.f});

    auto C = std::make_shared<Tensor2D>(2, 2);

    // Expected Result:
    // [ 58,  64 ]
    // [139, 154 ]

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((B->cols() + TILE_WIDTH - 1) / TILE_WIDTH,
                 (A->rows() + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled<<<dimGrid, dimBlock>>>(A->data(), B->data(), C->data(), A->rows(), A->cols(), B->cols());
    cudaDeviceSynchronize();

    REQUIRE(approx_equal(C->data()[0], 58.0f));
    REQUIRE(approx_equal(C->data()[1], 64.0f));
    REQUIRE(approx_equal(C->data()[2], 139.0f));
    REQUIRE(approx_equal(C->data()[3], 154.0f));
}

TEST_CASE("Kernel: Add Bias (Tensor2D)", "[kernel][bias]") {
    // Matrix (1x2)
    // 10 20
    auto Mat = std::make_shared<Tensor2D>(1, 2);
    Mat->set_data({10.f, 20.f});

    // Bias (1x2) - Broadcasts across rows
    // [1, 2]
    auto Bias = std::make_shared<Tensor2D>(1, 2);
    Bias->set_data({1.f, 2.f});

    unsigned int threads = 256;
    unsigned int blocks = (Bias->size() + threads - 1) / threads;
    vecadd_basic<<<blocks, threads>>>(Mat->data(), Bias->data(), Mat->data(), Mat->cols());

    /*
    unsigned int threads = 256;
    unsigned int blocks = (Bias->size() + threads - 1) / threads;
    vecadd_cascaded<<<blocks, threads>>>(Mat->data(), Bias->data(), Mat->data(), Mat->cols());
    */

    /*
    unsigned int threads = 256;
    unsigned int blocks = ((Bias->size() / 4) + threads - 1) / threads;
    assert((Bias->size() / 4) != 0);
    vecadd_cascaded<<<blocks, threads>>>(Mat->data(), Bias->data(), Mat->data(), Mat->cols());
    */

    cudaDeviceSynchronize();

    // Row 0: 10+1, 20+2 -> 11, 22
    REQUIRE(Mat->data()[0] == 11.0f);
    REQUIRE(Mat->data()[1] == 22.0f);
}
