//
// Created by felix on 09.04.19.
//

#ifndef CNN_GPU_CUDNN_NETWORK_H
#define CNN_GPU_CUDNN_NETWORK_H

#include <cudnn.h>
#include <cublas_v2.h>
#include "layers/FullyConnectedLayer.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/MaxPoolingLayer.h"

class Network{
private:
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
            conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_fc1, &ref_fc2;

    size_t SetFwdConvolutionTensors(ConvolutionalLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                             cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                             cudnnConvolutionFwdAlgo_t& algo);

    size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo);

public:
    Network(int gpuid, int batch_size,
                    ConvolutionalLayer& conv1, MaxPoolingLayer& pool1, ConvolutionalLayer& conv2, MaxPoolingLayer& pool2,
                    FullyConnectedLayer& fc1, FullyConnectedLayer& fc2);
    ~Network();

    void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                                     float *fc2, float *result,
                                     float *pconv1, float *pconv1bias,
                                     float *pconv2, float *pconv2bias,
                                     float *pfc1, float *pfc1bias,
                                     float *pfc2, float *pfc2bias, void *workspace, float *onevec);

    void Backpropagation(ConvolutionalLayer& layer_conv1, MaxPoolingLayer& layer_pool1, ConvolutionalLayer& layer_conv2, MaxPoolingLayer& layer_pool2,
                         float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                         float *fc2, float *fc2smax, float *dloss_data,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2,
                         void *workspace, float *onevec);

    void UpdateWeights(float learning_rate,
                       ConvolutionalLayer& conv1, ConvolutionalLayer& conv2,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias);

    int getBatchSize() const;

    size_t getWorkspaceSize() const;
};

#endif //CNN_GPU_CUDNN_NETWORK_H
