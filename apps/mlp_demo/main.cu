#include <nnlib/layers.h>
#include <nnlib/utils.h>

#include <iostream>
#include <algorithm>
#include <iomanip>

/* argmax: helper to find the index of the maximum value (ArgMax) */
int argmax(const float* data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    for(int i = 1; i < size; ++i) {
        if(data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char* argv[]) {
    try {
        std::string weights_file = "mnist_weights.bin";
        std::string samples_file = "mnist_samples.bin";

        /* load Model Weights */
        auto raw_tensors = load_binary_weights(weights_file);
        
        /* validation: 784->128->10 requires 4 tensors (W1, B1, W2, B2) */
        if(raw_tensors.size() < 4) {
            std::cerr << "Error: Model weights invalid." << std::endl;
            return 1;
        }

        /* Define architecture (must match what we defined during Python training!) */
        Sequential model;

        /* layer 1: 784 -> 128 */
        auto fc1 = std::make_shared<Linear>(784, 128);
        fc1->load_weights(raw_tensors[0], raw_tensors[1]);
        model.add(fc1);

        /* activation */
        model.add(std::make_shared<ReLU>());

        /* layer 2: 128 -> 10 */
        auto fc2 = std::make_shared<Linear>(128, 10);
        fc2->load_weights(raw_tensors[2], raw_tensors[3]);
        model.add(fc2);

        /* load test data */
        auto test_data = load_mnist_samples(samples_file);

        /* run inference loop */
        int correct = 0;
        int total = test_data.size();
        
        /* pre-allocate input tensor (batch size 1, 784 features) */
        auto input_tensor = std::make_shared<Tensor2D>(1, 784);

        std::cout << "\nStarting Inference on " << total << " images...\n" << std::endl;

        for(int i = 0; i < total; ++i) {
            /* load data into tensor (Unified Memory) */
            input_tensor->set_data(test_data[i].pixels);

            /* forward pass */
            auto output = model.forward(input_tensor);
            auto out_ptr = std::dynamic_pointer_cast<Tensor2D>(output);

            /* get prediction (argmax of the 10 outputs) */
            int prediction = argmax(out_ptr->data(), 10);
            int label = test_data[i].label;

            if (prediction == label) correct++;

            /* print first 10 to see it working */
            if(i < 10) {
                std::cout << "Img " << i << " | Pred: " << prediction 
                          << " | Actual: " << label 
                          << (prediction == label ? " [OK]" : " [FAIL]") << std::endl;
            }
        }

        /* final stats */
        float accuracy = (float)correct / total * 100.0f;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
