#ifndef BIASNEURON_H
#define BIASNEURON_H

#include "AbstractNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Neuron {

            class BiasNeuron : public Impulse::NeuralNetwork::Neuron::AbstractNeuron {
            public:

                BiasNeuron() : Impulse::NeuralNetwork::Neuron::AbstractNeuron(1) {
                }

                double forward(Eigen::VectorXd input) {
                    return this->weights(0);
                }
            };

        }

    }

}

#endif /* BIASNEURON_H */

