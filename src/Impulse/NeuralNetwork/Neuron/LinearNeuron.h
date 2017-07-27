#ifndef NEURON_H
#define NEURON_H

#include "AbstractNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Neuron {

            class LinearNeuron : public Impulse::NeuralNetwork::Neuron::AbstractNeuron {
            public:

                LinearNeuron(unsigned int size) : Impulse::NeuralNetwork::Neuron::AbstractNeuron(size) {
                    this->initializeWeights();
                    this->initializeDeltas();
                }

                void initializeWeights() {
                    double epsilon = 0.12;
                    for (unsigned int i = 0; i < this->size; i++) {
                        this->weights(i) = (((double) rand() / (RAND_MAX)) * 2 * epsilon - epsilon);
                    }
                }

                void initializeDeltas() {
                    this->deltas.setZero();
                }

                double forward(Eigen::VectorXd input) {
                    return input.dot(this->weights);
                }
            };

        }

    }

}

#endif /* NEURON_H */

