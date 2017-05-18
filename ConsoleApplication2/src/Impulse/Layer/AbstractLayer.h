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
    TypeVector * a;
    TypeVector * z;

public:

    AbstractLayer(int size, int prevSize) {
        this->size = size + 1;
        this->prevSize = prevSize;
        this->neurons = new NeuronContainer;
        this->a = new TypeVector;
        this->z = new TypeVector;
    };

    ~AbstractLayer() {
        for (auto i = this->neurons->begin(); i != this->neurons->end(); i++) {
            delete *i;
        }
        this->neurons->clear();
        delete this->neurons;
        
        this->a->clear();
        delete this->a;
        
        this->z->clear();
        delete this->z;
    }

    int getSize() {
        return this->size;
    }

    TypeVector * getA() {
        return this->a;
    }

    void reset() {
        this->a->clear();
        this->z->clear();
    }

    NeuronContainer * getNeurons() {
        return this->neurons;
    }

    void calculateDeltas(TypeVector * prevLayerA, TypeVector sigma) {
        for (int i = 0; i < sigma.size(); i++) {
            for (int j = 0; j < prevLayerA->size(); j++) {
                this->neurons->at(i + 1)->deltas->at(j) += prevLayerA->at(j) * sigma.at(i);
            }
        }
    }

    TypeVector backward(TypeVector sigma, AbstractLayer * nextLayer) {
        TypeVector resultSigma;
        for (int i = 0; i < this->size; i++) {
            resultSigma.push_back(0.0);
        }

        NeuronContainer * neurons = nextLayer->getNeurons();
        for (int i = 1; i < nextLayer->getSize(); i++) {
            TypeVector tmp = neurons->at(i)->backward(sigma.at(i - 1));
            for (int j = 0; j < tmp.size(); j++) {
                resultSigma.at(j) += tmp.at(j);
            }
        }
        resultSigma.erase(resultSigma.begin());

        TypeVector * a = this->getA();
        for (int i = 0; i < resultSigma.size(); i++) {
            resultSigma.at(i) *= a->at(i + 1) * (1.0 - a->at(i + 1));
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
                gradient.at(i - 1).push_back((this->neurons->at(i)->deltas->at(j) / (double) numSamples) + penalty.at(i - 1).at(j));
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

    virtual TypeVector forward(TypeVector input) = 0;
};

#endif /* LAYER_H */

