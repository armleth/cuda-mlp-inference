#pragma once

#include <vector>
#include <string>

struct MnistSample {
    std::vector<float> pixels; /* 28x28 image, 784 floats */
    int label;                 /* 0-9 */
};

std::vector<MnistSample> load_mnist_samples(const std::string& filename);

std::vector<std::vector<float>> load_binary_weights(const std::string& filename);
