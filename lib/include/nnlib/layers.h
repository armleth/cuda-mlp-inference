#pragma once

#include <nnlib/tensor.h>

#include <vector>
#include <memory>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
};

class Linear : public Layer {
    std::shared_ptr<Tensor2D> weights;
    std::shared_ptr<Tensor2D> bias;

public:
    Linear(int in_dim, int out_dim);
    void load_weights(const std::vector<float>& w, const std::vector<float>& b);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class ReLU : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Sequential : public Layer {
    std::vector<std::shared_ptr<Layer>> layers;
public:
    void add(std::shared_ptr<Layer> layer);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
