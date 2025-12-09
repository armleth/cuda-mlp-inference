add_executable(run_mlp main.cu)

target_link_libraries(run_mlp PRIVATE NNLib)

file(COPY ${CMAKE_SOURCE_DIR}/data/mnist_weights.bin DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CMAKE_SOURCE_DIR}/data/mnist_samples.bin DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
