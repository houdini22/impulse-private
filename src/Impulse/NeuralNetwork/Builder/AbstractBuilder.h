#ifndef IMPULSE_ABSTRACTBUILDER2_H
#define IMPULSE_ABSTRACTBUILDER2_H

#include "../Network/Network.h"
#include "../Layer/InputLayer.h"
#include "../Layer/OutputLayer.h"
#include "../../../Vendor/json.hpp"

using json = nlohmann::json;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class AbstractBuilder {
            protected:
                Impulse::NeuralNetwork::Network::Network *network;
                unsigned int prevSize;
            public:
                AbstractBuilder() {
                    this->network = new Impulse::NeuralNetwork::Network::Network();
                }

                AbstractBuilder *createInputLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::InputLayer *layer = new Impulse::NeuralNetwork::Layer::InputLayer(
                            size);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                    return this;
                }

                AbstractBuilder *addOutputLayer() {
                    Impulse::NeuralNetwork::Layer::OutputLayer *layer = new Impulse::NeuralNetwork::Layer::OutputLayer();
                    this->network->addLayer(layer);
                    return this;
                }

                virtual void createHiddenLayer(unsigned int size) = 0;

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
                            this->createInputLayer(it.value());
                        } else {
                            this->createHiddenLayer(it.value());
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

                    this->network->setRolledTheta(theta, true);

                    return this->network;
                }
            };

        }

    }

}

#endif //IMPULSE_ABSTRACTBUILDER_H
