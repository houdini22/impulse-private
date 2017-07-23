#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "./AbstractLayer.h"
#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/InputNeuron.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class InputLayer : public Impulse::NeuralNetwork::Layer::AbstractLayer {
            public:

                InputLayer(unsigned int size) : Impulse::NeuralNetwork::Layer::AbstractLayer(size, 1) {
                    this->a.resize(this->size);
                    this->z.resize(this->size);
                    this->createNeurons();
                }

                void createNeurons() {
                    this->neurons.push_back(new Impulse::NeuralNetwork::Neuron::BiasNeuron());
                    for (unsigned int i = 0; i < this->size - 1; i++) {
                        this->neurons.push_back(new Impulse::NeuralNetwork::Neuron::InputNeuron());
                    }
                }

                Eigen::VectorXd forward(Eigen::VectorXd input) {
                    this->reset();
                    Eigen::VectorXd output(this->size);

                    // get value from bias neuron
                    double biasResult = this->neurons.at(0)->forward(input);

                    output(0) = biasResult;
                    this->a(0) = biasResult;
                    this->z(0) = biasResult;

                    // start from 1 not bias neuron
                    unsigned int i = 0; // key for input
                    for (NeuronContainer::iterator it = this->neurons.begin() + 1; it != this->neurons.end(); ++it) {
                        Eigen::VectorXd prepared(1);
                        prepared(0) = input(i);
                        double result = (*it)->forward(prepared);

                        output(i + 1) = result;
                        this->a(i + 1) = result;
                        this->z(i + 1) = result;

                        i++;
                    }
                    return output;
                }

                Eigen::VectorXd
                backward(Eigen::VectorXd sigma, Impulse::NeuralNetwork::Layer::AbstractLayer *nextLayer) {
                    return Eigen::VectorXd();
                }

                double derivative(double input) {
                    return 0.0;
                }
            };

        }

    }

}

#endif /* INPUTLAYER_H */

