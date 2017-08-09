#ifndef IMPULSE_NEURALNETWORK_BUILDER_LINEARBUILDER_H
#define IMPULSE_NEURALNETWORK_BUILDER_LINEARBUILDER_H

#include "./AbstractBuilder.h"
#include "../Layer/LogisticLayer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class ClassificationBuilder : public AbstractBuilder {
            public:
                void createHiddenLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::LogisticLayer *layer = new Impulse::NeuralNetwork::Layer::LogisticLayer(
                            size, this->prevSize);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                }
            };

        }

    }

}

#endif //IMPULSE_ADALINE_H
