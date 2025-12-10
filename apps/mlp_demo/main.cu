#include <nnlib/layers.h>
#include <nnlib/utils.h>

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

/* Helper class for RAII CUDA Event handling */
class CudaTimer {
    cudaEvent_t start, stop;
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() {
        cudaEventRecord(start);
    }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

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
        
        if(raw_tensors.size() < 4) {
            std::cerr << "Error: Model weights invalid." << std::endl;
            return 1;
        }

        /* Define architecture */
        Sequential model;

        /* layer 1: 784 -> 128 */
        auto fc1 = std::make_shared<Linear>(784, 128);
        fc1->load_weights(raw_tensors[0], raw_tensors[1]);
        model.add(fc1);

        /* activation */
        auto relu = std::make_shared<ReLU>();
        model.add(relu);

        /* layer 2: 128 -> 10 */
        auto fc2 = std::make_shared<Linear>(128, 10);
        fc2->load_weights(raw_tensors[2], raw_tensors[3]);
        model.add(fc2);

        /* load test data */
        auto test_data = load_mnist_samples(samples_file);

        int correct = 0;
        int total = test_data.size();
        
        auto input_tensor = std::make_shared<Tensor2D>(1, 784);

        /* ---- BENCHMARKING VARIABLES ---- */
        CudaTimer global_timer;
        CudaTimer layer_timer;
        
        float total_inference_time_ms = 0.0f;
        float total_fc1_time = 0.0f;
        float total_relu_time = 0.0f;
        float total_fc2_time = 0.0f;

        std::cout << "\nStarting Inference on " << total << " images...\n" << std::endl;

        /* Start Global Timer */
        global_timer.tic();

        for(int i = 0; i < total; ++i) {
            /* load data into tensor */
            input_tensor->set_data(test_data[i].pixels);

            /* Manual Forward Pass to time individual layers */
            /* Note: Usually model.forward() handles this loop. To benchmark layers,
               we must manually step through them or modify the Sequential class.
               Here we manually invoke them for granular timing. */

            // 1. FC1 Forward
            layer_timer.tic();
            auto out1 = fc1->forward(input_tensor);
            total_fc1_time += layer_timer.toc();

            // 2. ReLU Forward
            layer_timer.tic();
            auto out2 = relu->forward(out1);
            total_relu_time += layer_timer.toc();

            // 3. FC2 Forward
            layer_timer.tic();
            auto output = fc2->forward(out2);
            total_fc2_time += layer_timer.toc();
            
            auto out_ptr = std::dynamic_pointer_cast<Tensor2D>(output);

            /* get prediction */
            int prediction = argmax(out_ptr->data(), 10);
            int label = test_data[i].label;

            if (prediction == label) correct++;

            /* print first 10 */
            if(i < 10) {
                std::cout << "Img " << i << " | Pred: " << prediction 
                          << " | Actual: " << label 
                          << (prediction == label ? " [OK]" : " [FAIL]") << std::endl;
            }
        }
        
        /* End Global Timer */
        float global_duration = global_timer.toc();
        
        /* Calculate aggregate inference time (sum of layers) to compare vs global wall time */
        total_inference_time_ms = total_fc1_time + total_relu_time + total_fc2_time;

        /* final stats */
        float accuracy = (float)correct / total * 100.0f;
        
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "BENCHMARK RESULTS (" << total << " samples):" << std::endl;
        std::cout << "Total Global Time (incl. data loading/cpu): " << global_duration << " ms" << std::endl;
        std::cout << "Total Pure GPU Inference Time:              " << total_inference_time_ms << " ms" << std::endl;
        std::cout << "Average Inference per sample:               " << (total_inference_time_ms / total) << " ms" << std::endl;
        std::cout << "\nLayer Breakdown (Average per pass):" << std::endl;
        std::cout << "  FC1 (784->128): " << (total_fc1_time / total) << " ms" << std::endl;
        std::cout << "  ReLU:           " << (total_relu_time / total) << " ms" << std::endl;
        std::cout << "  FC2 (128->10):  " << (total_fc2_time / total) << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
