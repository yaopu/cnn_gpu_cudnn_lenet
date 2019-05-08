#include "MaxPoolingLayer.h"

MaxPoolingLayer::MaxPoolingLayer(int poolingSize, int stride){
    this->poolingSize = poolingSize;
    this->stride = stride;
}

int MaxPoolingLayer::getPoolingSize() const {
    return poolingSize;
}

int MaxPoolingLayer::getStride() const {
    return stride;
}
