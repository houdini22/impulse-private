#include "NetworkTrainer.h"
#include "../Math/Minimizer/Fmincg.h"

NetworkTrainer::NetworkTrainer(Network *net) {
    this->network = net;
}

Network *NetworkTrainer::getNetwork() {
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

CostGradientResult NetworkTrainer::cost(DataSet &dataSet) {
    Network *net = this->network;
    double sumErrors = 0.0;
    std::vector<MapSample> samples = dataSet.getSamples();

    // reset deltas
    for (LayerContainer::iterator it = net->getLayers()->begin() + 1;
         it != net->getLayers()->end() - 1; ++it) {
        for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1;
             it2 != (*it)->getNeurons()->end(); ++it2) {
            for (int i = 0; i < (*it2)->deltas.size(); i++) {
                (*it2)->deltas(i) = 0;
            }
        }
    }

    for (int i = 0; i < samples.size(); i++) {
        Eigen::VectorXd predictions = net->forward(samples.at(i)["input"]);
        Eigen::VectorXd output = samples.at(i)["output"];
        net->backward(predictions, output);
        for (int j = 0; j < predictions.size(); j++) {
            // sumErrors += pow(predictions(j) - output(j), 2.0);
            sumErrors += ((output(j) * log(predictions(j))) + ((1 - output(j)) * log(1 - predictions(j))));
        }
    }

    double errorPenalty = 0.0;
    TypeVector resultGradient;
    LayerContainer *layers = net->getLayers();

    for (int i = 1; i < net->getSize() - 1; i++) {
        Eigen::MatrixXd penalty = layers->at(i)->backwardPenalty(
                dataSet.getSize(), this->regularization);
        errorPenalty += layers->at(i)->errorPenalty();
        Eigen::MatrixXd gradient = layers->at(i)->calculateGradient(
                dataSet.getSize(), penalty);
        Impulse::Math::Matrix::rollMatrixToVector(gradient, resultGradient);
    }

    double errorRegularization = (this->regularization * errorPenalty)
                                 / (2 * (double) dataSet.getSize());
    // double error = (1.0 / (2.0 * (double) dataSet.getSize())) * sumErrors + errorRegularization;
    double error = (-1.0 / (double) dataSet.getSize()) * sumErrors + errorRegularization;

    CostGradientResult result;
    result.error = error;
    result.gradient = Eigen::Map<Eigen::VectorXd>(resultGradient.data(),
                                                  resultGradient.size());

    return result;
}

void NetworkTrainer::train(DataSet dataSet) {
    Fmincg minimizer;

    Network *network = this->network;
    Eigen::VectorXd theta = network->getRolledTheta();

    std::function<CostGradientResult(Eigen::VectorXd)> cf(
            [this, &dataSet](Eigen::VectorXd input) {
                this->network->setRolledTheta(input);
                return this->cost(dataSet);
            });

    this->network->setRolledTheta(minimizer.minimize(cf, theta, this->learningIterations, true));
}