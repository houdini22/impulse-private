#ifndef IMPULSE_CLASSIFICATIONTRAINER_H
#define IMPULSE_CLASSIFICATIONTRAINER_H

#include "AbstractTrainer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class ClassificationTrainer : public AbstractTrainer {
            public:
                ClassificationTrainer(Impulse::NeuralNetwork::Network *net) : AbstractTrainer(net) {

                }

                double errorForSample(double prediction, double output) {
                    return ((output * log(prediction)) + ((1.0 - output) * log(1.0 - prediction)));
                }

                double calculateOverallError(unsigned int size, double sumErrors, double errorRegularization) {
                    return (-1.0 / (double) size) * sumErrors + errorRegularization;
                }
            };

        }

    }

}

#endif //IMPULSE_CLASSIFICATIONTRAINER_H
