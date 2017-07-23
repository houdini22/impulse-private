#ifndef NEURON_H
#define NEURON_H

#include "AbstractNeuron.h"

class Neuron : public AbstractNeuron {
public:

    Neuron(int size) : AbstractNeuron(size) {
        this->initializeWeights();
        this->initializeDeltas();
    }

    void initializeWeights() {
        double epsilon = 0.12;
        for (int i = 0; i < this->size; i++) {
            (*this->weights)(i) = (((double) rand() / (RAND_MAX)) * 2 * epsilon - epsilon);
        }
    }

    void initializeDeltas() {
        for (int i = 0; i < this->size; i++) {
            (*this->deltas)(i) = 0.0;
        }
    }

    double forward(Eigen::VectorXd input) {
        double result = 0.0;
        for (int i = 0; i < input.size(); i++) {
            result += input(i) * (*this->weights)(i);
        }
        return result;
    }
};

#endif /* NEURON_H */

