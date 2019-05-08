//
// Created by felix on 09.04.19.
//

#ifndef CNN_GPU_CUDNN_FULLYCONNECTEDLAYER_H
#define CNN_GPU_CUDNN_FULLYCONNECTEDLAYER_H

#include <vector>

class FullyConnectedLayer {
private:
    int inputs, outputs;

    std::vector<float> pneurons, pbias;

public:
    FullyConnectedLayer(int inputs, int outputs);

    int getInputs() const;

    int getOutputs() const;

    std::vector<float> &getPneurons();

    std::vector<float> &getPbias();

};

#endif //CNN_GPU_CUDNN_FULLYCONNECTEDLAYER_H
