//
// Created by felix on 09.04.19.
//

#ifndef CNN_GPU_CUDNN_CONVOLUTIONALLAYER_H
#define CNN_GPU_CUDNN_CONVOLUTIONALLAYER_H

#include <vector>

class ConvolutionalLayer{
private:
    int inputChannels, outputChannels,
        filterSize,
        inputHeight, inputWidth,
        outputHeight, outputWidth;

    std::vector<float> pconv, pbias;

public:
    ConvolutionalLayer(int inputChannels, int outputChannels, int filterSize, int inputHeight, int inputWidth);

    int getInputChannels() const;

    int getOutputChannels() const;

    int getFilterSize() const;

    int getInputHeight() const;

    int getInputWidth() const;

    int getOutputHeight() const;

    int getOutputWidth() const;

    std::vector<float> &getPconv();

    std::vector<float> &getPbias();
};

#endif //CNN_GPU_CUDNN_CONVOLUTIONALLAYER_H
