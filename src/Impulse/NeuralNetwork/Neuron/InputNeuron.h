#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include "AbstractNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Neuron {

            class InputNeuron : public Impulse::NeuralNetwork::Neuron::AbstractNeuron {
            public:

                InputNeuron() : Impulse::NeuralNetwork::Neuron::AbstractNeuron(0) {
                }

                double forward(Eigen::VectorXd input) {
                    return input(0);
                }
            };

        }

    }

}

#endif /* INPUTNEURON_H */

