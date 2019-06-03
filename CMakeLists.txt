cmake_minimum_required(VERSION 3.12)
project(cnn_gpu_cudnn_lenet)

find_package(CUDA 6.5 REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

set(HEADER_FILES
        readubyte.h
        layers/ConvolutionalLayer.h
        layers/MaxPoolingLayer.h
        layers/FullyConnectedLayer.h
        Network.h
        )

set(SOURCE_FILES
        reference_test.cu
        readubyte.cpp
        layers/ConvolutionalLayer.cu
        layers/MaxPoolingLayer.cu
        layers/FullyConnectedLayer.cu
        Network.cu
        )

cuda_add_executable(cudnn_reference_test ${SOURCE_FILES} ${HEADER_FILES})
cuda_add_cublas_to_target(cudnn_reference_test)

target_link_libraries(cudnn_reference_test cudnn)