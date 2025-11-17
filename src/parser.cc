#include "./parser.h"

#include <fstream>
#include <vector>

#include "./mlp.h"

using json = nlohmann::json;

MLP load_mlp(const std::string &filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    json j;
    f >> j;

    MLP mlp;

    mlp.version = j.at("version").get<int>();
    mlp.model_type = j.at("model_type").get<std::string>();
    mlp.input_dim = j.at("input_dim").get<int>();
    mlp.output_dim = j.at("output_dim").get<int>();
    mlp.hidden_layers = j.at("hidden_layers").get<std::vector<int>>();

    mlp.weights =
        j.at("weights").get<std::vector<std::vector<std::vector<double>>>>();
    mlp.biases = j.at("biases").get<std::vector<std::vector<double>>>();

    return mlp;
}
