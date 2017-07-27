#ifndef NETWORKBUILDER_H
#define NETWORKBUILDER_H

#include "Network.h"
#include "../Layer/InputLayer.h"
#include "../Layer/LogisticLayer.h"
#include "../Layer/OutputLayer.h"
#include "../Layer/PurelinLayer.h"
#include "../../../Vendor/json.hpp"

using json = nlohmann::json;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            class NetworkBuilder {
            protected:
                Impulse::NeuralNetwork::Network::Network *network;
                unsigned int prevSize;
            public:

                NetworkBuilder() {
                    this->network = new Impulse::NeuralNetwork::Network::Network();
                }

                NetworkBuilder *addInputLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::InputLayer *layer = new Impulse::NeuralNetwork::Layer::InputLayer(
                            size);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                    return this;
                }

                Impulse::NeuralNetwork::Network::NetworkBuilder *addLogisticLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::LogisticLayer *layer = new Impulse::NeuralNetwork::Layer::LogisticLayer(
                            size, this->prevSize);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                    return this;
                }

                Impulse::NeuralNetwork::Network::NetworkBuilder *addPurelinLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::PurelinLayer *layer = new Impulse::NeuralNetwork::Layer::PurelinLayer(
                            size, this->prevSize);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                    return this;
                }

                Impulse::NeuralNetwork::Network::NetworkBuilder *addOutputLayer() {
                    Impulse::NeuralNetwork::Layer::OutputLayer *layer = new Impulse::NeuralNetwork::Layer::OutputLayer();
                    this->network->addLayer(layer);
                    return this;
                }

                Impulse::NeuralNetwork::Network::Network *getNetwork() {
                    return this->network;
                }

                Impulse::NeuralNetwork::Network::Network *buildFromJSON(std::string path) {
                    std::ifstream fileStream(path);
                    json jsonFile;
                    fileStream >> jsonFile;
                    fileStream.close();

                    json savedLayers = jsonFile["layers"];
                    unsigned int i = 0;
                    for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
                        if (i == 0) {
                            this->addInputLayer(it.value());
                        } else {
                            this->addLogisticLayer(it.value());
                        }
                        i++;
                    }

                    this->addOutputLayer();

                    unsigned int j = 0;
                    json savedTheta = jsonFile["neurons"];
                    Eigen::VectorXd theta(savedTheta.size());
                    for (auto it = savedTheta.begin(); it != savedTheta.end(); ++it) {
                        theta(j++) = it.value();
                    }

                    this->network->setRolledTheta(theta);

                    return this->network;
                }
            };
        }

    }

}

#endif /* NETWORKBUILDER_H */

