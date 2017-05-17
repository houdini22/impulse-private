#ifndef NETWORKBUILDER_H
#define NETWORKBUILDER_H

#include "Network.h"
#include "../Layer/InputLayer.h"
#include "../Layer/HiddenLayer.h"
#include "../Layer/OutputLayer.h"

class NetworkBuilder {
protected:
    Network * network;
    int prevSize;
public:

    NetworkBuilder() {
        this->network = new Network();
    }

    NetworkBuilder * addInputLayer(int size) {
        InputLayer * layer = new InputLayer(size);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder * addHiddenLayer(int size) {
        HiddenLayer * layer = new HiddenLayer(size, this->prevSize);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder * addOutputLayer(int size) {
        OutputLayer * layer = new OutputLayer(size, this->prevSize);
        this->network->addLayer(layer);
        return this;
    }

    Network * getNetwork() {
        return this->network;
    }

    void clear() {
        delete this->network;
    }
};

#endif /* NETWORKBUILDER_H */

