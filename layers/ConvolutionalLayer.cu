#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(int inputChannels, int outputChannels, int filterSize, int inputHeight, int inputWidth)
    : pconv(inputChannels * filterSize * filterSize * outputChannels), pbias(outputChannels){
    this->inputChannels = inputChannels;
    this->outputChannels = outputChannels;
    this->filterSize = filterSize;
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->outputWidth = inputWidth - filterSize + 1;
    this->outputHeight = inputHeight - filterSize + 1;
}

int ConvolutionalLayer::getInputChannels() const {
    return inputChannels;
}

int ConvolutionalLayer::getOutputChannels() const {
    return outputChannels;
}

int ConvolutionalLayer::getFilterSize() const {
    return filterSize;
}

int ConvolutionalLayer::getInputHeight() const {
    return inputHeight;
}

int ConvolutionalLayer::getInputWidth() const {
    return inputWidth;
}

int ConvolutionalLayer::getOutputHeight() const {
    return outputHeight;
}

int ConvolutionalLayer::getOutputWidth() const {
    return outputWidth;
}

std::vector<float> &ConvolutionalLayer::getPconv() {
    return pconv;
}

std::vector<float> &ConvolutionalLayer::getPbias() {
    return pbias;
}
