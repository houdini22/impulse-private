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
		this->z = new Eigen::VectorXd(this->size);
        this->createNeurons();
    }

    void createNeurons() {
        this->neurons->push_back(new BiasNeuron());
        for (int i = 0; i < this->size - 1; i++) { // size is already computed with bias neuron so -1
            this->neurons->push_back(new Neuron(this->prevSize));
        }
    }

    TypeVector forward(TypeVector input) {
        this->reset();

        TypeVector output;

        // get value from bias neuron
        double biasResult = this->neurons->at(0)->forward(input);

        output.push_back(biasResult); // save to output layer
        (*this->a)(0) = (biasResult); // save to activated values container

        // start from 1 not bias neuron
		int i = 1;
        for (NeuronContainer::iterator it = this->neurons->begin() + 1; it != this->neurons->end(); ++it) {
            double result = (*it)->forward(input);

            (*this->z)(i - 1) = result; // save to raw output values container

            double activated = this->activation(result);
            //save
            output.push_back(activated);
            (*this->a)(i) = activated;

			i++;
        }
        return output;
    }

    double activation(double input) {
        return 1.0 / (1.0 + exp(-input));
    }
};

#endif /* HIDDENLAYER_H */

