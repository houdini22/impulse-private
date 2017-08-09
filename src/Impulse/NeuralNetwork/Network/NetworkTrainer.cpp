#include "NetworkTrainer.h"
#include "../Math/Minimizer/Fmincg.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            NetworkTrainer::NetworkTrainer(Impulse::NeuralNetwork::Network::Network *net) {
                this->network = net;
            }

            Impulse::NeuralNetwork::Network::Network *NetworkTrainer::getNetwork() {
                return this->network;
            }

            NetworkTrainer *NetworkTrainer::setRegularization(double regularization) {
                this->regularization = regularization;
                return this;
            }

            NetworkTrainer *NetworkTrainer::setLearningIterations(unsigned int nb) {
                this->learningIterations = nb;
                return this;
            }

            Impulse::NeuralNetwork::Network::CostGradientResult NetworkTrainer::cost(Impulse::SlicedDataset *dataSet) {
                Impulse::NeuralNetwork::Network::Network *net = this->network;
                double sumErrors = 0.0;

                // reset deltas
                for (LayerContainer::iterator it = net->getLayers()->begin() + 1;
                     it != net->getLayers()->end() - 1; ++it) {
                    for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1;
                         it2 != (*it)->getNeurons()->end(); ++it2) {
                        (*it2)->deltas.setZero();
                    }
                }

                Eigen::MatrixXd *inputMatrix = dataSet->getInput();
                Eigen::MatrixXd *outputMatrix = dataSet->getOutput();

                for (int i = 0; i < dataSet->input.getSize(); i++) {
                    Eigen::VectorXd predictions = net->forward(inputMatrix->row(i));
                    Eigen::VectorXd output = outputMatrix->row(i);
                    net->backward(predictions, output);
                    for (int j = 0; j < predictions.size(); j++) {
                        // sumErrors += pow(predictions(j) - output(j), 2.0);
                        sumErrors += ((output(j) * log(predictions(j))) + ((1.0 - output(j)) * log(1.0 - predictions(j))));
                    }
                }

                double errorPenalty = 0.0;
                Impulse::Math::TypeVector resultGradient;
                LayerContainer *layers = net->getLayers();

                for (int i = 1; i < net->getSize() - 1; i++) {
                    Eigen::MatrixXd penalty = layers->at(i)->backwardPenalty(
                            dataSet->input.getSize(), this->regularization);
                    errorPenalty += layers->at(i)->errorPenalty();
                    Eigen::MatrixXd gradient = layers->at(i)->calculateGradient(
                            dataSet->input.getSize(), penalty);
                    Impulse::Math::Matrix::rollMatrixToVector(gradient, resultGradient);
                }

                double errorRegularization = (this->regularization * errorPenalty)
                                             / (2 * (double) dataSet->input.getSize());
                // double error = (1.0 / (2.0 * (double) dataSet->input.getSize())) * sumErrors + errorRegularization;
                double error = (-1.0 / (double) dataSet->input.getSize()) * sumErrors + errorRegularization;

                Impulse::NeuralNetwork::Network::CostGradientResult result;
                result.error = error;
                result.gradient = Eigen::Map<Eigen::VectorXd>(resultGradient.data(),
                                                              resultGradient.size());

                return result;
            }

            void NetworkTrainer::train(Impulse::SlicedDataset *dataSet) {
                Impulse::NeuralNetwork::Math::Minimizer::Fmincg minimizer;

                Impulse::NeuralNetwork::Network::Network *network = this->network;
                Eigen::VectorXd theta = network->getRolledTheta();

                std::function<Impulse::NeuralNetwork::Network::CostGradientResult(Eigen::VectorXd)> cf(
                        [this, &dataSet](Eigen::VectorXd input) {
                            this->network->setRolledTheta(input);
                            return this->cost(dataSet);
                        });

                this->network->setRolledTheta(minimizer.minimize(cf, theta, this->learningIterations, true));
            }

        }

    }

}