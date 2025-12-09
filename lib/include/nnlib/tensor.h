#pragma once

#include <vector>
#include <memory>
#include <iostream>

class Tensor {
protected:
    float* _data;
    size_t _size;

public:
    explicit Tensor(size_t size);
    virtual ~Tensor();

    float* data() const { return _data; }
    size_t size() const { return _size; }

    virtual std::vector<int> get_shape() const = 0;
};

class Tensor2D : public Tensor {
    int _rows;
    int _cols;

public:
    Tensor2D(int r, int c);

    std::vector<int> get_shape() const override { return {_rows, _cols}; }
    int rows() const { return _rows; }
    int cols() const { return _cols; }

    void set_data(const std::vector<float>& input);
};
