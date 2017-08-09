#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "AbstractLayer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class OutputLayer : public Impulse::NeuralNetwork::Layer::AbstractLayer {
            public:

                OutputLayer() : Impulse::NeuralNetwork::Layer::AbstractLayer(0, 0) {

                }

                Eigen::VectorXd forward(Eigen::VectorXd input) {
                    Eigen::VectorXd newResult(input.size() - 1);
                    for (unsigned int i = 0; i < input.size() - 1; i++) {
                        newResult(i) = input(i + 1);
                    }
                    return newResult;
                }

                double derivative(double input) {
                    return 0.0;
                }

                double activation(double input) {
                    return input;
                }

                Eigen::VectorXd backward(Eigen::VectorXd sigma, AbstractLayer *nextLayer) {
                    return Eigen::VectorXd();
                }
            };

        }

    }

}

#endif /* OUTPUTLAYER_H */

