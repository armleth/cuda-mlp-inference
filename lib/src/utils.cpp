#include <nnlib/utils.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

std::vector<MnistSample> load_mnist_samples(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open sample file");

    unsigned int num_samples = 0;
    unsigned int input_dim = 0;
    
    file.read(reinterpret_cast<char*>(&num_samples), sizeof(unsigned int));
    file.read(reinterpret_cast<char*>(&input_dim), sizeof(unsigned int));

    std::vector<MnistSample> samples;
    samples.reserve(num_samples);

    for(unsigned int i=0; i<num_samples; ++i) {
        MnistSample s;
        s.pixels.resize(input_dim);
        
        /* read pixels */
        file.read(reinterpret_cast<char*>(s.pixels.data()), input_dim * sizeof(float));
        
        /* read label */
        file.read(reinterpret_cast<char*>(&s.label), sizeof(int));
        
        samples.push_back(s);
    }
    std::cout << "Loaded " << samples.size() << " MNIST test images." << std::endl;
    return samples;
}

std::vector<std::vector<float>> load_binary_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary weights file: " + filename);
    }

    unsigned int num_tensors = 0;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(unsigned int));

    std::vector<std::vector<float>> all_tensors;
    all_tensors.reserve(num_tensors);

    for (unsigned int i = 0; i < num_tensors; ++i) {
        unsigned int num_elements = 0;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(unsigned int));

        std::vector<float> tensor_data(num_elements);
        file.read(reinterpret_cast<char*>(tensor_data.data()), num_elements * sizeof(float));
        
        if (!file) {
            throw std::runtime_error("Error reading tensor data at index " + std::to_string(i));
        }

        all_tensors.push_back(std::move(tensor_data));
    }

    std::cout << "Loaded " << num_tensors << " tensors from " << filename << std::endl;
    return all_tensors;
}
