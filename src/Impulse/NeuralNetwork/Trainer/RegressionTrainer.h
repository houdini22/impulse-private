#ifndef IMPULSE_REGRESSIONTRAINER_H
#define IMPULSE_REGRESSIONTRAINER_H

#include "AbstractTrainer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class RegressionTrainer : public AbstractTrainer {
            public:
                RegressionTrainer(Impulse::NeuralNetwork::Network::Network *net) : AbstractTrainer(net) {

                }
                double errorForSample(double prediction, double output) {
                    return pow(prediction - output, 2.0);
                }

                double calculateOverallError(unsigned int size, double sumErrors, double errorRegularization) {
                    return (1.0 / (2.0 * (double) size)) * sumErrors + errorRegularization;
                }
            };

        }

    }

}

#endif //IMPULSE_CLASSIFICATIONTRAINER_H
