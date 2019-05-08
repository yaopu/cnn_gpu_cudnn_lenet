#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int inputs, int outputs): pneurons(inputs * outputs), pbias(outputs){
    this->inputs = inputs;
    this->outputs = outputs;
}

int FullyConnectedLayer::getInputs() const {
    return inputs;
}

int FullyConnectedLayer::getOutputs() const {
    return outputs;
}

std::vector<float> &FullyConnectedLayer::getPneurons() {
    return pneurons;
}

std::vector<float> &FullyConnectedLayer::getPbias() {
    return pbias;
}
