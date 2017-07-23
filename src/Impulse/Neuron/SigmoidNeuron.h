#ifndef NEURON_H
#define NEURON_H

#include "AbstractNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Neuron {

            class SigmoidNeuron : public Impulse::NeuralNetwork::Neuron::AbstractNeuron {
            public:

                SigmoidNeuron(unsigned int size) : Impulse::NeuralNetwork::Neuron::AbstractNeuron(size) {
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
                    for (unsigned int i = 0; i < this->size; i++) {
                        this->deltas(i) = 0.0;
                    }
                }

                double forward(Eigen::VectorXd input) {
                    double result = input.dot(this->weights);
                    return result;
                }
            };

        }

    }

}

#endif /* NEURON_H */

