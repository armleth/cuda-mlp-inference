enable_testing()

add_executable(run_tests 
    main_test.cpp
    test_kernels.cu
)

target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(run_tests PRIVATE NNLib)

add_test(NAME AllTests COMMAND run_tests)
