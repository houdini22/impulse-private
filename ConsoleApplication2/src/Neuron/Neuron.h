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
            this->weights->push_back(((double) rand() / (RAND_MAX)) * 2 * epsilon - epsilon);
        }
    }

    void initializeDeltas() {
        for (int i = 0; i < this->size; i++) {
            this->deltas->push_back(0.0);
        }
    }

    double forward(TypeVector input) {
        double result = 0.0;
        for (int i = 0; i < input.size(); i++) {
            result += input.at(i) * this->weights->at(i);
        }
        return result;
    }
};

#endif /* NEURON_H */

