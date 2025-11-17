#pragma once

#include <string>
#include <vector>

struct MLP
{
    int version;
    std::string model_type;
    int input_dim;
    int output_dim;
    std::vector<int> hidden_layers;

    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
};
