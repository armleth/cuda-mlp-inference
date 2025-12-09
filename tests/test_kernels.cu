#include "catch.hpp"
#include <nnlib/kernels.h>
#include <nnlib/tensor.h>
#include <nnlib/matmul.h>
#include <vector>
#include <cmath>

/* approx_equal: helper for floating point comparison */
bool approx_equal(float a, float b, float epsilon = 0.001f) {
    return std::abs(a - b) < epsilon;
}

TEST_CASE("Kernel: ReLU Activation (Tensor2D)", "[kernel][relu]") {
    auto t = std::make_shared<Tensor2D>(1, 5);
    std::vector<float> input = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    t->set_data(input);

    launch_relu(t->data(), t->size());

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

    /* launch_matmul(A->data(), B->data(), C->data(), A->rows(), A->cols(), B->cols()); */
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
    // Matrix (2x2)
    // 10 20
    // 30 40
    auto Mat = std::make_shared<Tensor2D>(2, 2);
    Mat->set_data({10.f, 20.f, 30.f, 40.f});

    // Bias (1x2) - Broadcasts across rows
    // [1, 2]
    auto Bias = std::make_shared<Tensor2D>(1, 2);
    Bias->set_data({1.f, 2.f});

    launch_add_bias(Mat->data(), Bias->data(), Mat->rows(), Mat->cols());

    // Row 0: 10+1, 20+2 -> 11, 22
    // Row 1: 30+1, 40+2 -> 31, 42
    REQUIRE(Mat->data()[0] == 11.0f);
    REQUIRE(Mat->data()[1] == 22.0f);
    REQUIRE(Mat->data()[2] == 31.0f);
    REQUIRE(Mat->data()[3] == 42.0f);
}
