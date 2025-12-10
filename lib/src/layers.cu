#include <nnlib/layers.h>
#include <nnlib/vecadd.h>
#include <nnlib/matmul.h>
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

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((output_dim + TILE_WIDTH - 1) / TILE_WIDTH,
                 (batch_size + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled<<<dimGrid, dimBlock>>>(input2d->data(), weights->data(), output->data(), batch_size, input_dim, output_dim);
    cudaDeviceSynchronize();

    unsigned int threads = 256;
    unsigned int blocks = (bias->size() + threads - 1) / threads;
    vecadd_basic<<<threads, blocks>>>(output->data(), bias->data(), output->data(), output_dim); // not very usefull
    cudaDeviceSynchronize();

    return output;
}

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    unsigned int threads = 256;
    unsigned int blocks = (input->size() + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input->data(), input->size());
    cudaDeviceSynchronize();

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
