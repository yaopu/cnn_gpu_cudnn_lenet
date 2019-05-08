//
// Created by felix on 09.04.19.
//

#ifndef CNN_GPU_CUDNN_MAXPOOLINGLAYER_H
#define CNN_GPU_CUDNN_MAXPOOLINGLAYER_H

class MaxPoolingLayer{
private:
    int poolingSize;
    int stride;


public:
    MaxPoolingLayer(int poolingSize, int stride);

    int getPoolingSize() const;

    int getStride() const;
};

#endif //CNN_GPU_CUDNN_MAXPOOLINGLAYER_H
