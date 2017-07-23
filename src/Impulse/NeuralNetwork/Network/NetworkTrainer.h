#pragma once

#include <functional>
#include <string>
#include <stdio.h>
#include <memory>
#include <cstring>

#include "Network.h"
#include "../Data/DataSet.h"
#include "../Math/Matrix.h"
#include "../Math/Minimizer/Fmincg.h"

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

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

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

                CostGradientResult cost(DataSet &dataSet);

                void train(DataSet dataSet);
            };

        }

    }

}
