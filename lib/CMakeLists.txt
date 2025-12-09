add_library(NNLib
    src/kernels.cu
    src/tensor.cu
    src/layers.cpp
    src/utils.cpp
)

target_include_directories(NNLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(NNLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
