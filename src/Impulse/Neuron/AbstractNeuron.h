#ifndef ABSTRACTNEURON_H
#define ABSTRACTNEURON_H

#include <stdlib.h>
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include "../Math/Matrix.h"
#include "../Math/Vector.h"

class AbstractNeuron {
protected:
    int size;
public:
	Eigen::VectorXd * weights;
	Eigen::VectorXd * deltas;

    AbstractNeuron(int size) {
        this->size = size;
        this->weights = new Eigen::VectorXd(size);
        this->deltas = new Eigen::VectorXd(size);
    }

    ~AbstractNeuron() {
		this->weights->resize(0);
        delete this->weights;

		this->deltas->resize(0);
        delete this->deltas;
    }

    int getSize() {
        return this->size;
    }

	Eigen::VectorXd backward(double sigma) {
		Eigen::VectorXd result(*this->weights);
		result *= sigma;
        return result;
    }

    Eigen::VectorXd backwardPenalty(int numSamples, double regularization) {
		Eigen::VectorXd result(this->size);
        result(0) = 0.0;
        for (int i = 1; i < this->size; i++) {
            result(i) = ((regularization / (double) numSamples) * (*this->weights)(i));
        }
        return result;
    }

    double errorPenalty() {
        double sum = 0.0;
        for (int i = 1; i < this->size; i++) {
            sum += pow((*this->weights)(i), 2.0);
        }
        return sum;
    }

    virtual double forward(Eigen::VectorXd input) = 0;
};

#endif /* NEURON_H */

