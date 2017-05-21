#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include <math.h>

#include "./AbstractLayer.h"
#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/Neuron.h"

class HiddenLayer : public AbstractLayer {
public:

    HiddenLayer(int size, int prevSize) : AbstractLayer(size, prevSize) {
		this->a = new Eigen::VectorXd(this->size);
		this->z = new Eigen::VectorXd(this->size - 1);
        this->createNeurons();
    }

    void createNeurons() {
        this->neurons->push_back(new BiasNeuron());
        for (int i = 0; i < this->size - 1; i++) { // size is already computed with bias neuron so -1
            this->neurons->push_back(new Neuron(this->prevSize));
        }
    }

	Eigen::VectorXd backward(Eigen::VectorXd sigma, AbstractLayer * nextLayer) {
		Eigen::VectorXd tmpResultSigma(this->size);
		for (int i = 0; i < this->size; i++) {
			tmpResultSigma(i) = 0.0;
		}

		NeuronContainer * neurons = nextLayer->getNeurons();
		for (int i = 1; i < nextLayer->getSize(); i++) {
			Eigen::VectorXd tmp = neurons->at(i)->backward(sigma(i - 1));
			for (int j = 0; j < tmp.size(); j++) {
				tmpResultSigma(j) += tmp(j);
			}
		}

		Eigen::VectorXd resultSigma(this->size - 1);
		for (int i = 1; i < tmpResultSigma.size(); i++) {
			resultSigma(i - 1) = tmpResultSigma(i);
		}

		Eigen::VectorXd * a = this->getA();
		for (int i = 0; i < resultSigma.size(); i++) {
			resultSigma(i) *= this->derivative((*a)(i + 1));
		}

		return resultSigma;
	}

	Eigen::VectorXd forward(Eigen::VectorXd input) {
        this->reset();

		Eigen::VectorXd output(this->neurons->size());

        // get value from bias neuron
        double biasResult = this->neurons->at(0)->forward(input);

        output(0) = biasResult; // save to output layer
        (*this->a)(0) = (biasResult); // save to activated values container

        // start from 1 not bias neuron
		int i = 1;
        for (NeuronContainer::iterator it = this->neurons->begin() + 1; it != this->neurons->end(); ++it) {
            double result = (*it)->forward(input);

            (*this->z)(i - 1) = result; // save to raw output values container

            double activated = this->activation(result);
            //save
            output(i) = activated;
            (*this->a)(i) = activated;

			i++;
        }
        return output;
    }

	double derivative(double input) {
		return input * (1.0 - input);
	}

    double activation(double input) {
        return 1.0 / (1.0 + exp(-input));
    }
};

#endif /* HIDDENLAYER_H */

