
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "readubyte.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/MaxPoolingLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "Network.h"
#include <chrono>

using namespace std::chrono;

// Block width for CUDA kernels
#define BLOCK_WIDTH 128

#define TIME_MEASURE true

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

int main()
{
#if TIME_MEASURE
    size_t sumForwardMicroseconds = 0;
    size_t sumBackwardMicroseconds = 0;
    size_t averageForwardMicroseconds = 0;
    size_t averageBackwardMicroseconds = 0;
#endif

    size_t width, height, channels = 1;
    size_t BATCH_SIZE = 64;
    size_t ITERATIONS = 15;

    // Open input data
    printf("Reading input data\n");
    
    // Read dataset sizes
    size_t train_size = ReadUByteDataset("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", nullptr, nullptr, width, height);
    size_t test_size = ReadUByteDataset("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", nullptr, nullptr, width, height);
    if (train_size == 0)
        return 1;
    
    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", &train_images[0], &train_labels[0], width, height) != train_size)
        return 2;
    if (ReadUByteDataset("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", &test_images[0], &test_labels[0], width, height) != test_size)
        return 3;

    printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    printf("Image size: %dx%dx%d, Batch size: %d, iterations: %d\n",width, height, channels, BATCH_SIZE, ITERATIONS);

    // Choose GPU
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    int gpuId = 0;
    if (gpuId < 0 || gpuId >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               gpuId, num_gpus);
        return 4;
    }

    printf("\nInstalled Devices (%d):\n", num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if(i == gpuId){
            printf("  Device Number: %d  [SELECTED]\n", i);
        }else{
            printf("  Device Number: %d\n", i);
        }
        printf("    Device name: %s\n", prop.name);
        printf("    Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("    Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("    Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    // Create the network architecture
    ConvolutionalLayer conv1((int)channels, 6, 5, (int)width, (int)height);
    MaxPoolingLayer pool1(2, 2);
    ConvolutionalLayer conv2(conv1.getOutputChannels(), 16, 5, conv1.getOutputWidth() / pool1.getStride(), conv1.getOutputHeight() / pool1.getStride());
    MaxPoolingLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.getOutputChannels()*conv2.getOutputWidth()*conv2.getOutputHeight()) / (pool2.getStride() * pool2.getStride()),
                            120);
    FullyConnectedLayer fc2(fc1.getOutputs(), 84);

    // Initialize CUDNN/CUBLAS network
    Network network(gpuId, BATCH_SIZE, conv1, pool1, conv2, pool2, fc1, fc2);

    // Create random network
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier weight filling
    float wconv1 = sqrt(3.0f / (conv1.getFilterSize() * conv1.getFilterSize() * conv1.getInputChannels()));
    std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
    float wconv2 = sqrt(3.0f / (conv2.getFilterSize() * conv2.getFilterSize() * conv2.getInputChannels()));
    std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
    float wfc1 = sqrt(3.0f / (fc1.getInputs() * fc1.getOutputs()));
    std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
    float wfc2 = sqrt(3.0f / (fc2.getInputs() * fc2.getOutputs()));
    std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

    // Randomize network
    for (auto&& iter : conv1.getPconv())
        iter = static_cast<float>(dconv1(gen));
    for (auto&& iter : conv1.getPbias())
        iter = static_cast<float>(dconv1(gen));
    for (auto&& iter : conv2.getPconv())
        iter = static_cast<float>(dconv2(gen));
    for (auto&& iter : conv2.getPbias())
        iter = static_cast<float>(dconv2(gen));
    for (auto&& iter : fc1.getPneurons())
        iter = static_cast<float>(dfc1(gen));
    for (auto&& iter : fc1.getPbias())
        iter = static_cast<float>(dfc1(gen));
    for (auto&& iter : fc2.getPneurons())
        iter = static_cast<float>(dfc2(gen));
    for (auto&& iter : fc2.getPbias())
        iter = static_cast<float>(dfc2(gen));

    
    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures    

    // Forward propagation data
    float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    cudaMalloc(&d_data,    sizeof(float) * network.getBatchSize() * channels           * height                            * width);
    cudaMalloc(&d_labels,  sizeof(float) * network.getBatchSize() * 1                  * 1                                 * 1);
    cudaMalloc(&d_conv1,   sizeof(float) * network.getBatchSize() * conv1.getOutputChannels() * conv1.getOutputHeight()                  * conv1.getOutputWidth());
    cudaMalloc(&d_pool1,   sizeof(float) * network.getBatchSize() * conv1.getOutputChannels() * (conv1.getOutputHeight() / pool1.getStride()) * (conv1.getOutputWidth() / pool1.getStride()));
    cudaMalloc(&d_conv2,   sizeof(float) * network.getBatchSize() * conv2.getOutputChannels() * conv2.getOutputHeight()                  * conv2.getOutputWidth());
    cudaMalloc(&d_pool2,   sizeof(float) * network.getBatchSize() * conv2.getOutputChannels() * (conv2.getOutputHeight() / pool2.getStride()) * (conv2.getOutputWidth() / pool2.getStride()));
    cudaMalloc(&d_fc1,     sizeof(float) * network.getBatchSize() * fc1.getOutputs());
    cudaMalloc(&d_fc1relu, sizeof(float) * network.getBatchSize() * fc1.getOutputs());
    cudaMalloc(&d_fc2,     sizeof(float) * network.getBatchSize() * fc2.getOutputs());
    cudaMalloc(&d_fc2smax, sizeof(float) * network.getBatchSize() * fc2.getOutputs());


    // Network parameters
    float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
    
    cudaMalloc(&d_pconv1,     sizeof(float) * conv1.getPconv().size());
    cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.getPbias().size());
    cudaMalloc(&d_pconv2,     sizeof(float) * conv2.getPconv().size());
    cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.getPbias().size());
    cudaMalloc(&d_pfc1,       sizeof(float) * fc1.getPneurons().size());
    cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.getPbias().size());
    cudaMalloc(&d_pfc2,       sizeof(float) * fc2.getPneurons().size());
    cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.getPbias().size());

    
    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    cudaMalloc(&d_gconv1,     sizeof(float) * conv1.getPconv().size());
    cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.getPbias().size());
    cudaMalloc(&d_gconv2,     sizeof(float) * conv2.getPconv().size());
    cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.getPbias().size());
    cudaMalloc(&d_gfc1,       sizeof(float) * fc1.getPneurons().size());
    cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.getPbias().size());
    cudaMalloc(&d_gfc2,       sizeof(float) * fc2.getPneurons().size());
    cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.getPbias().size());
    
    
    // Differentials w.r.t. data
    float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    cudaMalloc(&d_dpool1,   sizeof(float) * network.getBatchSize() * conv1.getOutputChannels() * conv1.getOutputHeight()                  * conv1.getOutputWidth());
    cudaMalloc(&d_dpool2,   sizeof(float) * network.getBatchSize() * conv2.getOutputChannels() * conv2.getOutputHeight()                  * conv2.getOutputWidth());
    cudaMalloc(&d_dconv2,   sizeof(float) * network.getBatchSize() * conv1.getOutputChannels() * (conv1.getOutputHeight() / pool1.getStride()) * (conv1.getOutputWidth() / pool1.getStride()));
    cudaMalloc(&d_dfc1,     sizeof(float) * network.getBatchSize() * fc1.getInputs());
    cudaMalloc(&d_dfc1relu, sizeof(float) * network.getBatchSize() * fc1.getOutputs());
    cudaMalloc(&d_dfc2,     sizeof(float) * network.getBatchSize() * fc2.getInputs());
    cudaMalloc(&d_dfc2smax, sizeof(float) * network.getBatchSize() * fc2.getOutputs());
    cudaMalloc(&d_dlossdata,sizeof(float) * network.getBatchSize() * fc2.getOutputs());
    

    // Temporary buffers and workspaces
    float *d_onevec;
    void *d_cudnn_workspace = nullptr;    
    cudaMalloc(&d_onevec, sizeof(float)* network.getBatchSize());
    if (network.getWorkspaceSize() > 0)
        cudaMalloc(&d_cudnn_workspace, network.getWorkspaceSize());

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
    cudaMemcpyAsync(d_pconv1, &conv1.getPconv()[0],     sizeof(float) * conv1.getPconv().size(),  cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pconv1bias, &conv1.getPbias()[0], sizeof(float) * conv1.getPbias().size(),  cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pconv2, &conv2.getPconv()[0],     sizeof(float) * conv2.getPconv().size(),  cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pconv2bias, &conv2.getPbias()[0], sizeof(float) * conv2.getPbias().size(),  cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pfc1, &fc1.getPneurons()[0],      sizeof(float) * fc1.getPneurons().size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pfc1bias, &fc1.getPbias()[0],     sizeof(float) * fc1.getPbias().size(),    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pfc2, &fc2.getPneurons()[0],      sizeof(float) * fc2.getPneurons().size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_pfc2bias, &fc2.getPbias()[0],     sizeof(float) * fc2.getPbias().size(),    cudaMemcpyHostToDevice);


    // Fill one-vector with ones
    FillOnes<<<RoundUp(network.getBatchSize(), BLOCK_WIDTH), BLOCK_WIDTH>>>(d_onevec, network.getBatchSize());

    printf("Preparing dataset\n");
    
    // Normalize training set to be in [0,1]
    std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        train_images_float[i] = (float)train_images[i] / 255.0f;
    
    for (size_t i = 0; i < train_size; ++i)
        train_labels_float[i] = (float)train_labels[i];

    printf("Training...\n");

    // Use SGD to train the network
    cudaDeviceSynchronize();
    //auto t1 = std::chrono::high_resolution_clock::now();

    size_t numBatches = train_size/network.getBatchSize();
    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        for(int batchNr = 0; batchNr < numBatches; batchNr++) {
            // Train
            //size_t imageid = iter % (train_size / network.getBatchSize());

            // Prepare current batch on device
            cudaMemcpyAsync(d_data, &train_images_float[batchNr * network.getBatchSize() * width * height * channels],
                            sizeof(float) * network.getBatchSize() * channels * width * height, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(d_labels, &train_labels_float[batchNr * network.getBatchSize()],
                            sizeof(float) * network.getBatchSize(), cudaMemcpyHostToDevice);
#if TIME_MEASURE
            high_resolution_clock::time_point t_forward_start = high_resolution_clock::now();
#endif

            // Forward propagation
            network.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                       d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2,
                                       d_pfc2bias,
                                       d_cudnn_workspace, d_onevec);
#if TIME_MEASURE
            high_resolution_clock::time_point t_forward_stop = high_resolution_clock::now();
            sumForwardMicroseconds += duration_cast<microseconds>(t_forward_stop-t_forward_start).count();


            high_resolution_clock::time_point t_backward_start = high_resolution_clock::now();
#endif
            // Backward propagation
            network.Backpropagation(conv1, pool1, conv2, pool2,
                                    d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2,
                                    d_fc2smax, d_dlossdata,
                                    d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2,
                                    d_pfc2bias,
                                    d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2,
                                    d_gfc1, d_gfc1bias,
                                    d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

            // Compute learning rate
            float learningRate = static_cast<float>(0.01 /*learning rate*/ *
                                                    pow((1.0 + 0.0001 /*gamma*/ * iter), (-0.75 /*power*/)));

            // Update weights
            network.UpdateWeights(learningRate, conv1, conv2,
                                  d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2,
                                  d_pfc2bias,
                                  d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2,
                                  d_gfc2bias);
#if TIME_MEASURE
            high_resolution_clock::time_point t_backward_stop = high_resolution_clock::now();
            sumBackwardMicroseconds += duration_cast<microseconds>(t_backward_stop-t_backward_start).count();
#endif

        }
    }
    cudaDeviceSynchronize();
    //auto t2 = std::chrono::high_resolution_clock::now();

    //printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);

#if TIME_MEASURE
    averageForwardMicroseconds = sumForwardMicroseconds / (ITERATIONS*numBatches);
    averageBackwardMicroseconds = sumBackwardMicroseconds / (ITERATIONS*numBatches);
#endif

    std::cout << "Average Forward duration in microseconds: " << averageForwardMicroseconds << std::endl;
    std::cout << "Average Backward duration in microseconds: " << averageBackwardMicroseconds << std::endl;


    float classification_error = 1.0f;
    int classifications = (int)test_size;
    
    // Test the resulting neural network's classification
    if (classifications > 0)
    {
        // Initialize a TrainingContext structure for testing (different batch size)
        Network test_network(gpuId, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        if (network.getWorkspaceSize() < test_network.getWorkspaceSize())
        {
            cudaFree(d_cudnn_workspace);
            cudaMalloc(&d_cudnn_workspace, test_network.getWorkspaceSize());
        }

        int num_errors = 0;
        for (int i = 0; i < classifications; ++i)
        {
            std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            for (int j = 0; j < width * height; ++j)
                data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice);
            
            // Forward propagate test image
            test_network.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                            d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);

            // Perform classification
            std::vector<float> class_vec(10);

            // Copy back result
            cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost);

            // Determine classification according to maximal response
            int chosen = 0;
            for (int id = 1; id < 10; ++id)
            {
                if (class_vec[chosen] < class_vec[id]) chosen = id;
            }

            if (chosen != test_labels[i])
                ++num_errors;
        }
        classification_error = (float)num_errors / (float)classifications;

        printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
    }
        
    // Free data structures
    cudaFree(d_data);
    cudaFree(d_conv1);
    cudaFree(d_pool1);
    cudaFree(d_conv2);
    cudaFree(d_pool2);
    cudaFree(d_fc1);
    cudaFree(d_fc2);
    cudaFree(d_pconv1);
    cudaFree(d_pconv1bias);
    cudaFree(d_pconv2);
    cudaFree(d_pconv2bias);
    cudaFree(d_pfc1);
    cudaFree(d_pfc1bias);
    cudaFree(d_pfc2);
    cudaFree(d_pfc2bias);
    cudaFree(d_gconv1);
    cudaFree(d_gconv1bias);
    cudaFree(d_gconv2);
    cudaFree(d_gconv2bias);
    cudaFree(d_gfc1);
    cudaFree(d_gfc1bias);
    cudaFree(d_dfc1);
    cudaFree(d_gfc2);
    cudaFree(d_gfc2bias);
    cudaFree(d_dfc2);
    cudaFree(d_dpool1);
    cudaFree(d_dconv2);
    cudaFree(d_dpool2);
    cudaFree(d_labels);
    cudaFree(d_dlossdata);
    cudaFree(d_onevec);
    if (d_cudnn_workspace != nullptr)
        cudaFree(d_cudnn_workspace);

    return 0;
}
