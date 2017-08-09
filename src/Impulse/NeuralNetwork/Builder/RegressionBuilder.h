#ifndef IMPULSE_NEURALNETWORK_BUILDER_LOGISTICBUILDER_H
#define IMPULSE_NEURALNETWORK_BUILDER_LOGISTICBUILDER_H

#include "AbstractBuilder.h"
#include "../Layer/PurelinLayer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class RegressionBuilder : public Impulse::NeuralNetwork::Builder::AbstractBuilder {
            public:
                void createHiddenLayer(unsigned int size) {
                    Impulse::NeuralNetwork::Layer::PurelinLayer *layer = new Impulse::NeuralNetwork::Layer::PurelinLayer(
                            size, this->prevSize);
                    this->network->addLayer(layer);
                    this->prevSize = layer->getSize();
                }
            };

        }

    }

}

#endif //IMPULSE_ADALINE_H
