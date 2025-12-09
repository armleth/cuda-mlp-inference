#include <nnlib/tensor.h>

#include <stdexcept>
#include <algorithm>

Tensor::Tensor(size_t size) : _size(size) {
    cudaError_t err = cudaMallocManaged(&_data, _size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Malloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

Tensor::~Tensor() {
    if (_data) {
        cudaFree(_data);
    }
}

Tensor2D::Tensor2D(int r, int c) : Tensor(r * c), _rows(r), _cols(c) {}

void Tensor2D::set_data(const std::vector<float>& input) {
    if (input.size() != _size) {
        std::cerr << "Error: Input size mismatch. Expected " << _size 
                  << ", got " << input.size() << std::endl;
        return;
    }
    std::copy(input.begin(), input.end(), _data);
}
