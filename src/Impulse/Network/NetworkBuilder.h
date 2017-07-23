#ifndef NETWORKBUILDER_H
#define NETWORKBUILDER_H

#include "Network.h"
#include "../Layer/InputLayer.h"
#include "../Layer/HiddenLayer.h"
#include "../Layer/OutputLayer.h"
#include "../Layer/PurelinLayer.h"
#include "../../Vendor/json.hpp"

using json = nlohmann::json;

class NetworkBuilder {
protected:
    Network *network;
    int prevSize;
public:

    NetworkBuilder() {
        this->network = new Network();
    }

    NetworkBuilder *addInputLayer(int size) {
        Impulse::NeuralNetwork::Layer::InputLayer *layer = new Impulse::NeuralNetwork::Layer::InputLayer(size);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder *addHiddenLayer(int size) {
        Impulse::NeuralNetwork::Layer::HiddenLayer *layer = new Impulse::NeuralNetwork::Layer::HiddenLayer(size, this->prevSize);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder *addHiddenPurelinLayer(int size) {
        Impulse::NeuralNetwork::Layer::PurelinLayer *layer = new Impulse::NeuralNetwork::Layer::PurelinLayer(size, this->prevSize);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder *addOutputLayer() {
        Impulse::NeuralNetwork::Layer::OutputLayer *layer = new Impulse::NeuralNetwork::Layer::OutputLayer();
        this->network->addLayer(layer);
        return this;
    }

    Network *getNetwork() {
        return this->network;
    }

    void clear() {
        delete this->network;
    }

    Network *buildFromJSON(std::string path) {
        std::ifstream fileStream(path);
        json jsonFile;
        fileStream >> jsonFile;
        fileStream.close();

        json savedLayers = jsonFile["layers"];
        int i = 0;
        for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
            if (i == 0) {
                this->addInputLayer(it.value());
            } else {
                this->addHiddenPurelinLayer(it.value());
            }
            i++;
        }

        this->addOutputLayer();

        int j = 0;
        json savedTheta = jsonFile["neurons"];
        Eigen::VectorXd theta(savedTheta.size());
        for (auto it = savedTheta.begin(); it != savedTheta.end(); ++it) {
            theta(j++) = it.value();
        }

        this->network->setRolledTheta(theta);

        return this->network;
    }
};

#endif /* NETWORKBUILDER_H */

