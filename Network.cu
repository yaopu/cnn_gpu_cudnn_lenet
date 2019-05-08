#include "Network.h"

// Block width for CUDA kernels
#define BLOCK_WIDTH 128

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

Network::Network(int gpuId, int batchSize,
        ConvolutionalLayer& conv1, MaxPoolingLayer& pool1, ConvolutionalLayer& conv2, MaxPoolingLayer& pool2,
        FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuId) {
    m_batchSize = batchSize;

    // Create CUBLAS and CUDNN handles
    cudaSetDevice(gpuId);
    cublasCreate(&cublasHandle);
    cudnnCreate(&cudnnHandle);

    // Create tensor descriptors
    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&conv1Tensor);
    cudnnCreateTensorDescriptor(&conv1BiasTensor);
    cudnnCreateTensorDescriptor(&pool1Tensor);
    cudnnCreateTensorDescriptor(&conv2Tensor);
    cudnnCreateTensorDescriptor(&conv2BiasTensor);
    cudnnCreateTensorDescriptor(&pool2Tensor);
    cudnnCreateTensorDescriptor(&fc1Tensor);
    cudnnCreateTensorDescriptor(&fc2Tensor);

    cudnnCreateActivationDescriptor(&fc1Activation);

    cudnnCreateFilterDescriptor(&conv1filterDesc);
    cudnnCreateFilterDescriptor(&conv2filterDesc);

    cudnnCreateConvolutionDescriptor(&conv1Desc);
    cudnnCreateConvolutionDescriptor(&conv2Desc);

    cudnnCreatePoolingDescriptor(&poolDesc);


    // Set tensor descriptor sizes
    cudnnSetTensor4dDescriptor(conv1BiasTensor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1, conv1.getOutputChannels(),
                               1, 1);
    cudnnSetTensor4dDescriptor(conv2BiasTensor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1, conv2.getOutputChannels(),
                               1, 1);

    cudnnSetPooling2dDescriptor(poolDesc,
                                CUDNN_POOLING_MAX,
                                CUDNN_PROPAGATE_NAN,
                                pool1.getPoolingSize(), pool1.getPoolingSize(),
                                0, 0,
                                pool1.getStride(), pool1.getStride());
    cudnnSetTensor4dDescriptor(pool2Tensor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batchSize, conv2.getOutputChannels(),
                               conv2.getOutputHeight() / pool2.getStride(),
                               conv2.getOutputWidth() / pool2.getStride());

    cudnnSetTensor4dDescriptor(fc1Tensor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batchSize, fc1.getOutputs(), 1, 1);

    cudnnSetTensor4dDescriptor(fc2Tensor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batchSize, fc2.getOutputs(), 1, 1);

    cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                 CUDNN_PROPAGATE_NAN, 0.0);

    // Set convolution tensor sizes and compute workspace size
    size_t workspace = 0;
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

    workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

    // The workspace is allocated later (if necessary)
    m_workspaceSize = workspace;
}

Network::~Network()
{
    cudaSetDevice(m_gpuid);

    cublasDestroy(cublasHandle);
    cudnnDestroy(cudnnHandle);
    cudnnDestroyTensorDescriptor(dataTensor);
    cudnnDestroyTensorDescriptor(conv1Tensor);
    cudnnDestroyTensorDescriptor(conv1BiasTensor);
    cudnnDestroyTensorDescriptor(pool1Tensor);
    cudnnDestroyTensorDescriptor(conv2Tensor);
    cudnnDestroyTensorDescriptor(conv2BiasTensor);
    cudnnDestroyTensorDescriptor(pool2Tensor);
    cudnnDestroyTensorDescriptor(fc1Tensor);
    cudnnDestroyTensorDescriptor(fc2Tensor);
    cudnnDestroyActivationDescriptor(fc1Activation);
    cudnnDestroyFilterDescriptor(conv1filterDesc);
    cudnnDestroyFilterDescriptor(conv2filterDesc);
    cudnnDestroyConvolutionDescriptor(conv1Desc);
    cudnnDestroyConvolutionDescriptor(conv2Desc);
    cudnnDestroyPoolingDescriptor(poolDesc);
}

void Network::ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                        float *fc2, float *result,
                        float *pconv1, float *pconv1bias,
                        float *pconv2, float *pconv2bias,
                        float *pfc1, float *pfc1bias,
                        float *pfc2, float *pfc2bias, void *workspace, float *onevec)
{
    float alpha = 1.0f, beta = 0.0f;
    cudaSetDevice(m_gpuid);

    // Conv1 layer
    cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                            data, conv1filterDesc, pconv1, conv1Desc,
                            conv1algo, workspace, m_workspaceSize, &beta,
                            conv1Tensor, conv1);
    cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                   pconv1bias, &alpha, conv1Tensor, conv1);

    // Pool1 layer
    cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                        conv1, &beta, pool1Tensor, pool1);

    // Conv2 layer
    cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                            pool1, conv2filterDesc, pconv2, conv2Desc,
                            conv2algo, workspace, m_workspaceSize, &beta,
                            conv2Tensor, conv2);
    cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                   pconv2bias, &alpha, conv2Tensor, conv2);

    // Pool2 layer
    cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                        conv2, &beta, pool2Tensor, pool2);

    // FC1 layer
    // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                ref_fc1.getOutputs(), m_batchSize, ref_fc1.getInputs(),
                &alpha,
                pfc1, ref_fc1.getInputs(),
                pool2, ref_fc1.getInputs(),
                &beta,
                fc1, ref_fc1.getOutputs());
    // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                ref_fc1.getOutputs(), m_batchSize, 1,
                &alpha,
                pfc1bias, ref_fc1.getOutputs(),
                onevec, 1,
                &alpha,
                fc1, ref_fc1.getOutputs());

    // ReLU activation
    cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                           fc1Tensor, fc1, &beta, fc1Tensor, fc1relu);

    // FC2 layer
    // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                ref_fc2.getOutputs(), m_batchSize, ref_fc2.getInputs(),
                &alpha,
                pfc2, ref_fc2.getInputs(),
                fc1relu, ref_fc2.getInputs(),
                &beta,
                fc2, ref_fc2.getOutputs());
    // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                ref_fc2.getOutputs(), m_batchSize, 1,
                &alpha,
                pfc2bias, ref_fc2.getOutputs(),
                onevec, 1,
                &alpha,
                fc2, ref_fc2.getOutputs());

    // Softmax loss
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result);
}

void Network::Backpropagation(ConvolutionalLayer& layer_conv1, MaxPoolingLayer& layer_pool1, ConvolutionalLayer& layer_conv2, MaxPoolingLayer& layer_pool2,
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
                     void *workspace, float *onevec)
{
    float alpha = 1.0f, beta = 0.0f;

    float scalVal = 1.0f / static_cast<float>(m_batchSize);

    cudaSetDevice(m_gpuid);

    // Initialization (using the training error function)
    cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.getOutputs(), cudaMemcpyDeviceToDevice);

    // Softmax layer
    SoftmaxLossBackprop<<<RoundUp(m_batchSize, BLOCK_WIDTH), BLOCK_WIDTH>>>(labels, ref_fc2.getOutputs(), m_batchSize, dloss_data);

    // Accounting for batch size in SGD
    cublasSscal(cublasHandle, ref_fc2.getOutputs() * m_batchSize, &scalVal, dloss_data, 1);

    // FC2 layer
    // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.getInputs(), ref_fc2.getOutputs(), m_batchSize,
                &alpha, fc1relu, ref_fc2.getInputs(), dloss_data, ref_fc2.getOutputs(), &beta, gfc2, ref_fc2.getInputs());
    // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
    cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.getOutputs(), m_batchSize,
                &alpha, dloss_data, ref_fc2.getOutputs(), onevec, 1, &beta, gfc2bias, 1);
    // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.getInputs(), m_batchSize, ref_fc2.getOutputs(),
                &alpha, pfc2, ref_fc2.getInputs(), dloss_data, ref_fc2.getOutputs(), &beta, dfc2, ref_fc2.getInputs());

    // ReLU activation
    cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                            fc1Tensor, fc1relu, fc1Tensor, dfc2,
                            fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu);

    // FC1 layer
    // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.getInputs(), ref_fc1.getOutputs(), m_batchSize,
                &alpha, pool2, ref_fc1.getInputs(), dfc1relu, ref_fc1.getOutputs(), &beta, gfc1, ref_fc1.getInputs());
    // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
    cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.getOutputs(), m_batchSize,
                &alpha, dfc1relu, ref_fc1.getOutputs(), onevec, 1, &beta, gfc1bias, 1);
    // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.getInputs(), m_batchSize, ref_fc1.getOutputs(),
                &alpha, pfc1, ref_fc1.getInputs(), dfc1relu, ref_fc1.getOutputs(), &beta, dfc1, ref_fc1.getInputs());

    // Pool2 layer
    cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                         pool2Tensor, pool2, pool2Tensor, dfc1,
                         conv2Tensor, conv2, &beta, conv2Tensor, dpool2);

    // Conv2 layer
    cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                 dpool2, &beta, conv2BiasTensor, gconv2bias);


    cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                   pool1, conv2Tensor, dpool2, conv2Desc,
                                   conv2bwfalgo, workspace, m_workspaceSize,
                                   &beta, conv2filterDesc, gconv2);

    cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                 pconv2, conv2Tensor, dpool2, conv2Desc,
                                 conv2bwdalgo, workspace, m_workspaceSize,
                                 &beta, pool1Tensor, dconv2);

    // Pool1 layer
    cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                         pool1Tensor, pool1, pool1Tensor, dconv2,
                         conv1Tensor, conv1, &beta, conv1Tensor, dpool1);

    // Conv1 layer
    cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                 dpool1, &beta, conv1BiasTensor, gconv1bias);

    cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                   data, conv1Tensor, dpool1, conv1Desc,
                                   conv1bwfalgo, workspace, m_workspaceSize,
                                   &beta, conv1filterDesc, gconv1);

    // No need for convBackwardData because there are no more layers below
}

void Network::UpdateWeights(float learning_rate,
                   ConvolutionalLayer& conv1, ConvolutionalLayer& conv2,
                   float *pconv1, float *pconv1bias,
                   float *pconv2, float *pconv2bias,
                   float *pfc1, float *pfc1bias,
                   float *pfc2, float *pfc2bias,
                   float *gconv1, float *gconv1bias,
                   float *gconv2, float *gconv2bias,
                   float *gfc1, float *gfc1bias,
                   float *gfc2, float *gfc2bias)
{
    float alpha = -learning_rate;

    cudaSetDevice(m_gpuid);

    // Conv1
    cublasSaxpy(cublasHandle, static_cast<int>(conv1.getPconv().size()),
                &alpha, gconv1, 1, pconv1, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(conv1.getPbias().size()),
                &alpha, gconv1bias, 1, pconv1bias, 1);

    // Conv2
    cublasSaxpy(cublasHandle, static_cast<int>(conv2.getPconv().size()),
                &alpha, gconv2, 1, pconv2, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(conv2.getPbias().size()),
                &alpha, gconv2bias, 1, pconv2bias, 1);

    // Fully connected 1
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.getPneurons().size()),
                &alpha, gfc1, 1, pfc1, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.getPbias().size()),
                &alpha, gfc1bias, 1, pfc1bias, 1);

    // Fully connected 2
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.getPneurons().size()),
                &alpha, gfc2, 1, pfc2, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.getPbias().size()),
                &alpha, gfc2bias, 1, pfc2bias, 1);
}

size_t Network::SetFwdConvolutionTensors(ConvolutionalLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                cudnnConvolutionFwdAlgo_t& algo)
{
    size_t sizeInBytes = 0;

    int n = m_batchSize;
    int c = conv.getInputChannels();
    int h = conv.getInputHeight();
    int w = conv.getInputWidth();

    cudnnSetTensor4dDescriptor(srcTensorDesc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               n, c,
                               h, w);

    cudnnSetFilter4dDescriptor(filterDesc,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               conv.getOutputChannels(),
                               conv.getInputChannels(),
                               conv.getFilterSize(),
                               conv.getFilterSize());

#if CUDNN_MAJOR > 5
    cudnnSetConvolution2dDescriptor(convDesc,
                                    0, 0,
                                    1, 1,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
#else
    cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION);
#endif

    // Find dimension of convolution output
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          srcTensorDesc,
                                          filterDesc,
                                          &n, &c, &h, &w);

    cudnnSetTensor4dDescriptor(dstTensorDesc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               n, c,
                               h, w);
    cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                        srcTensorDesc,
                                        filterDesc,
                                        convDesc,
                                        dstTensorDesc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0,
                                        &algo);

    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                            srcTensorDesc,
                                            filterDesc,
                                            convDesc,
                                            dstTensorDesc,
                                            algo,
                                            &sizeInBytes);

    return sizeInBytes;
}

size_t Network::SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                         cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                         cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
{
    size_t sizeInBytes = 0, tmpsize = 0;

    // If backprop filter algorithm was requested
    if (falgo)
    {
        cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo);

        cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                *falgo, &tmpsize);

        sizeInBytes = std::max(sizeInBytes, tmpsize);
    }

    // If backprop data algorithm was requested
    if (dalgo)
    {
        cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo);

        cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                *dalgo, &tmpsize);

        sizeInBytes = std::max(sizeInBytes, tmpsize);
    }

    return sizeInBytes;
}

int Network::getBatchSize() const {
    return m_batchSize;
}

size_t Network::getWorkspaceSize() const {
    return m_workspaceSize;
}
