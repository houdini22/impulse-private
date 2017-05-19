#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>
#include "../Neuron/AbstractNeuron.h"
#include "../Math/Matrix.h"
#include "../Math/Vector.h"

typedef std::vector<AbstractNeuron *> NeuronContainer;

class AbstractLayer {
protected:
    int size;
    int prevSize;
    NeuronContainer * neurons;
	Eigen::VectorXd * a;
	Eigen::VectorXd * z;

public:

    AbstractLayer(int size, int prevSize) {
        this->size = size + 1;
        this->prevSize = prevSize;
		this->neurons = new NeuronContainer;
    };

    ~AbstractLayer() {
        for (auto i = this->neurons->begin(); i != this->neurons->end(); i++) {
            delete *i;
        }
        this->neurons->clear();
        delete this->neurons;
        
		this->a->resize(0);
        delete this->a;
        
		this->z->resize(0);
        delete this->z;
    }

    int getSize() {
        return this->size;
    }

	Eigen::VectorXd * getA() {
        return this->a;
    }

    void reset() {
        
    }

    NeuronContainer * getNeurons() {
        return this->neurons;
    }

    void calculateDeltas(Eigen::VectorXd * prevLayerA, Eigen::VectorXd sigma) {
        for (int i = 0; i < sigma.size(); i++) {
            for (int j = 0; j < prevLayerA->size(); j++) {
                (*this->neurons->at(i + 1)->deltas)(j) += (*prevLayerA)(j) * sigma(i);
            }
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
            resultSigma(i) *= (*a)(i + 1) * (1.0 - (*a)(i + 1));
        }

        return resultSigma;
    }

    TypeMatrix backwardPenalty(int numSamples, double regularization) {
        TypeMatrix penalty;
        for (int i = 1; i < this->size; i++) {
            penalty.push_back(this->neurons->at(i)->backwardPenalty(numSamples, regularization));
        }
        return penalty;
    }

    TypeMatrix calculateGradient(int numSamples, TypeMatrix penalty) {
        TypeMatrix gradient;
        for (int i = 1; i < this->size; i++) {
            gradient.push_back(TypeVector());
            for (int j = 0; j < this->neurons->at(i)->deltas->size(); j++) {
                gradient.at(i - 1).push_back(((*this->neurons->at(i)->deltas)(j) / (double) numSamples) + penalty.at(i - 1).at(j));
            }
        }
        return gradient;
    }

    double errorPenalty() {
        double sum = 0.0;
        for (int i = 1; i < this->size; i++) {
            sum += this->neurons->at(i)->errorPenalty();
        }
        return sum;
    }

    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
};

#endif /* LAYER_H */

