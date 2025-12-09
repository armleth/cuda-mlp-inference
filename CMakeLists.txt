cmake_minimum_required(VERSION 3.18)
project(CudaNNInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(lib)

add_subdirectory(apps/mlp_demo)
# add_subdirectory(apps/cnn_demo)

add_subdirectory(tests)
