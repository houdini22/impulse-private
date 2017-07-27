#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>
#include "../Neuron/AbstractNeuron.h"
#include "../Math/Matrix.h"
#include "../Math/Vector.h"

typedef std::vector<Impulse::NeuralNetwork::Neuron::AbstractNeuron *> NeuronContainer;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class AbstractLayer {
            protected:
                unsigned int size;
                unsigned int prevSize;
                NeuronContainer neurons;
                Eigen::VectorXd a;
                Eigen::VectorXd z;

            public:

                AbstractLayer(unsigned int size, unsigned int prevSize) {
                    this->size = size + 1;
                    this->prevSize = prevSize;
                };

                ~AbstractLayer() {
                    for (auto i = this->neurons.begin(); i != this->neurons.end(); i++) {
                        delete *i;
                    }
                    this->neurons.clear();
                    this->a.resize(0);
                    this->z.resize(0);
                }

                unsigned int getSize() {
                    return this->size;
                }

                Eigen::VectorXd *getA() {
                    return &this->a;
                }

                NeuronContainer *getNeurons() {
                    return &this->neurons;
                }

                void calculateDeltas(Eigen::VectorXd *prevLayerA, Eigen::VectorXd sigma) {
                    for (unsigned int i = 0; i < sigma.size(); i++) {
                        for (unsigned int j = 0; j < prevLayerA->size(); j++) {
                            this->neurons.at(i + 1)->deltas(j) += (*prevLayerA)(j) * sigma(i);
                        }
                    }
                }

                Eigen::MatrixXd backwardPenalty(unsigned int numSamples, double regularization) {
                    Eigen::MatrixXd resultPenalty(this->size - 1, this->prevSize);
                    for (unsigned int i = 1; i < this->size; i++) {
                        resultPenalty.row(i - 1) = this->neurons.at(i)->backwardPenalty(numSamples, regularization);
                    }
                    return resultPenalty;
                }

                Eigen::MatrixXd calculateGradient(unsigned int numSamples, Eigen::MatrixXd penalty) {
                    Eigen::MatrixXd gradient(this->neurons.at(1)->getSize(), this->size - 1);
                    for (unsigned int i = 1; i < this->size; i++) {
                        for (unsigned int j = 0; j < this->neurons.at(i)->deltas.size(); j++) {
                            gradient(j, i - 1) =
                                    (this->neurons.at(i)->deltas(j) / (double) numSamples) + penalty(i - 1, j);
                        }
                    }
                    return gradient;
                }

                double errorPenalty() {
                    double sum = 0.0;
                    for (unsigned int i = 1; i < this->size; i++) {
                        sum += this->neurons.at(i)->errorPenalty();
                    }
                    return sum;
                }

                virtual double derivative(double input) = 0;

                virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;

                virtual Eigen::VectorXd
                backward(Eigen::VectorXd sigma, Impulse::NeuralNetwork::Layer::AbstractLayer *nextLayer) = 0;
            };

        }

    }

}

#endif /* LAYER_H */

