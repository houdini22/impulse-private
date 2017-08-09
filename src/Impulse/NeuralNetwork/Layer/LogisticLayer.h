#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include <math.h>

#include "./AbstractLayer.h"
#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/LinearNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class LogisticLayer : public Impulse::NeuralNetwork::Layer::AbstractLayer {
            public:

                LogisticLayer(unsigned int size, unsigned int prevSize) : Impulse::NeuralNetwork::Layer::AbstractLayer(
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
                    return input * (1.0 - input);
                }

                double activation(double input) {
                    return 1.0 / (1.0 + exp(-input));
                }
            };

        }

    }

}

#endif /* HIDDENLAYER_H */

