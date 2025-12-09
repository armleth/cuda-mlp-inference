#include <nnlib/layers.h>
#include <nnlib/kernels.h>

#include <stdexcept>

Linear::Linear(int in_dim, int out_dim) {
    weights = std::make_shared<Tensor2D>(in_dim, out_dim);
    bias = std::make_shared<Tensor2D>(1, out_dim);
}

void Linear::load_weights(const std::vector<float>& w, const std::vector<float>& b) {
    weights->set_data(w);
    bias->set_data(b);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto input2d = std::dynamic_pointer_cast<Tensor2D>(input);
    if (!input2d) throw std::runtime_error("Linear layer requires Tensor2D input");

    int batch_size = input2d->rows();
    int input_dim = input2d->cols();
    int output_dim = weights->cols();

    auto output = std::make_shared<Tensor2D>(batch_size, output_dim);

    launch_matmul(input2d->data(), weights->data(), output->data(), 
                  batch_size, input_dim, output_dim);

    launch_add_bias(output->data(), bias->data(), batch_size, output_dim);

    return output;
}

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    launch_relu(input->data(), input->size());
    return input;
}

void Sequential::add(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    auto current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}
