#ifndef PURELINLAYER_H
#define PURELINLAYER_H

#include "AbstractLayer.h"
#include "../Neuron/LinearNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class PurelinLayer : public Impulse::NeuralNetwork::Layer::AbstractLayer {
            public:

                PurelinLayer(unsigned int size, unsigned int prevSize) : Impulse::NeuralNetwork::Layer::AbstractLayer(
                        size, prevSize) {
                    this->a.resize(this->size);
                    this->createNeurons();
                }

                void createNeurons() {
                    this->neurons.push_back(new Impulse::NeuralNetwork::Neuron::BiasNeuron());
                    for (unsigned int i = 0;
                         i < this->size - 1; i++) { // size is already computed with bias neuron so -1
                        this->neurons.push_back(new Impulse::NeuralNetwork::Neuron::LinearNeuron(this->prevSize));
                    }
                }

                double derivative(double input) {
                    return 1.0;
                }

                double activation(double input) {
                    return input;
                }
            };

        }

    }

}

#endif /* OUTPUTLAYER_H */

