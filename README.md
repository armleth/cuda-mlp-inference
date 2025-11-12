# CUDA and Deep Learning – Inference Only

The goal of this project is to implement a neural network using CUDA (inference only).

## Level 1 – MLP

### Implement CUDA Primitive Functions

Using CUDA, implement the following basic operations:
- Matrix addition
- Matrix multiplication
- feedforward_layer
- ReLU activation

### Implement a Forward Layer

Implement the forward pass of a single neural network layer.
You can load weights directly from a file.

### Implement the MLP Forward Pass

Implement the forward pass of a full MLP.
Weights should be pre-trained and loaded from a format of your choice (e.g. text file, ONNX file, etc.).

### Benchmark Inference Speed

Compare inference speed between:
- Your CUDA implementation
- PyTorch
- (Optionally) ONNX Runtime

Test on multiple architecture sizes.

### Optimize

Apply any optimizations that come to mind to improve speed or efficiency.

## Level 2 – CNN

Repeat Level 1 for convolutional neural networks (CNNs).
Implement the necessary CUDA kernels for convolution, activation, and pooling operations.

## Level 3 – Advanced Extensions

Implement one or more of the following ideas:

A more complex model of your choice (e.g. YOLO, RNN, small Transformer, etc.)

Implement Level 1 and/or Level 2 using Triton
