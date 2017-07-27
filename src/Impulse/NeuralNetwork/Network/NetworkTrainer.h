#pragma once

#include <functional>
#include <string>
#include <stdio.h>
#include <memory>
#include <cstring>

#include "Network.h"
#include "../Math/Matrix.h"
#include "../Math/Minimizer/Fmincg.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            struct CostGradientResult {
                double error;
                Eigen::VectorXd gradient;

                double getCost() {
                    return this->error;
                }

                Eigen::VectorXd getGradient() {
                    return this->gradient;
                }
            };

            class NetworkTrainer {
            protected:
                Impulse::NeuralNetwork::Network::Network *network;
                double regularization = 0.0;
                unsigned int learningIterations = 1;
            public:

                NetworkTrainer(Impulse::NeuralNetwork::Network::Network *net);

                Impulse::NeuralNetwork::Network::Network *getNetwork();

                Impulse::NeuralNetwork::Network::NetworkTrainer *setRegularization(double regularization);

                Impulse::NeuralNetwork::Network::NetworkTrainer *setLearningIterations(unsigned int nb);

                Impulse::NeuralNetwork::Network::CostGradientResult cost(Impulse::SlicedDataset *dataSet);

                void train(Impulse::SlicedDataset *dataSet);
            };

        }

    }

}
