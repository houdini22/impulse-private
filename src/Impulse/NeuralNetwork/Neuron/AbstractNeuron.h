#ifndef ABSTRACTNEURON_H
#define ABSTRACTNEURON_H

#include <stdlib.h>
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include "../Math/Matrix.h"
#include "../Math/Vector.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Neuron {

            class AbstractNeuron {
            protected:
                unsigned int size;
            public:
                Eigen::VectorXd weights;
                Eigen::VectorXd deltas;

                AbstractNeuron(unsigned int size) {
                    this->size = size;
                    this->weights.resize(size);
                    this->deltas.resize(size);
                }

                ~AbstractNeuron() {
                    this->weights.resize(0);
                    this->deltas.resize(0);
                }

                unsigned int getSize() {
                    return this->size;
                }

                Eigen::VectorXd backward(double sigma) {
                    Eigen::VectorXd result(this->weights);
                    result *= sigma;
                    return result;
                }

                Eigen::VectorXd backwardPenalty(unsigned int numSamples, double regularization) {
                    Eigen::VectorXd result(this->size);
                    result(0) = 0.0;
                    for (unsigned int i = 1; i < this->size; i++) {
                        result(i) = ((regularization / (double) numSamples) * this->weights(i));
                    }
                    return result;
                }

                double errorPenalty() {
                    double sum = 0.0;
                    for (unsigned int i = 1; i < this->size; i++) {
                        sum += pow(this->weights(i), 2.0);
                    }
                    return sum;
                }

                virtual double forward(Eigen::VectorXd input) = 0;
            };

        }

    }

}

#endif /* NEURON_H */

