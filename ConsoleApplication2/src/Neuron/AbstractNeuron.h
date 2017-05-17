#ifndef ABSTRACTNEURON_H
#define ABSTRACTNEURON_H

#include <stdlib.h>
#include <vector>
#include <complex>
#include "../Math/Matrix.h"
#include "../Math/Vector.h"

class AbstractNeuron {
protected:
    int size;
public:
    TypeVector * weights;
    TypeVector * deltas;

    AbstractNeuron(int size) {
        this->size = size;
        this->weights = new TypeVector;
        this->deltas = new TypeVector;
    }

    ~AbstractNeuron() {
        this->weights->clear();
        delete this->weights;

        this->deltas->clear();
        delete this->deltas;
    }

    int getSize() {
        return this->size;
    }

    TypeVector backward(double sigma) {
        TypeVector result;
        for (int i = 0; i < this->weights->size(); i++) {
            result.push_back(sigma * this->weights->at(i));
        }
        return result;
    }

    TypeVector backwardPenalty(int numSamples, double regularization) {
        TypeVector result;
        result.push_back(0.0);
        for (int i = 1; i < this->size; i++) {
            result.push_back((regularization / (double) numSamples) * this->weights->at(i));
        }
        return result;
    }

    double errorPenalty() {
        double sum = 0.0;
        for (int i = 1; i < this->size; i++) {
            sum += pow(this->weights->at(i), 2.0);
        }
        return sum;
    }

    virtual double forward(TypeVector input) = 0;
};

#endif /* NEURON_H */

