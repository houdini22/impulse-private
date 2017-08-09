#ifndef IMPULSE_ABSTRACTTRAINER_H
#define IMPULSE_ABSTRACTTRAINER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../Network/Network.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

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

            class AbstractTrainer {
            protected:
                Impulse::NeuralNetwork::Network::Network *network;
                double regularization = 0.0;
                unsigned int learningIterations = 1;
            public:
                AbstractTrainer(Impulse::NeuralNetwork::Network::Network *net);

                Impulse::NeuralNetwork::Network::Network *getNetwork();

                void setRegularization(double regularization);

                void setLearningIterations(unsigned int nb);

                CostGradientResult cost(Impulse::SlicedDataset *dataSet);

                void train(Impulse::SlicedDataset *dataSet);

                virtual double errorForSample(double prediction, double output) = 0;

                virtual double calculateOverallError(unsigned int size, double sumErrors, double errorRegularization) = 0;
            };

        }

    }

}

#endif //IMPULSE_ABSTRACTTRAINER_H
