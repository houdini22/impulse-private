#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "./AbstractLayer.h"
#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/InputNeuron.h"

class InputLayer : public AbstractLayer {
public:

    InputLayer(int size) : AbstractLayer(size, 1) {
		this->a = new Eigen::VectorXd(this->size);
		this->z = new Eigen::VectorXd(this->size);
        this->createNeurons();
    }

    void createNeurons() {
        this->neurons->push_back(new BiasNeuron());
        for (int i = 0; i < this->size - 1; i++) {
            this->neurons->push_back(new InputNeuron());
        }
    }

    TypeVector forward(TypeVector input) {
        this->reset();
        TypeVector output;

        // get value from bias neuron
        double biasResult = this->neurons->at(0)->forward(input);

        output.push_back(biasResult);
        (*this->a)(0) = biasResult;
        (*this->z)(0) = biasResult;

        // start from 1 not bias neuron
        int i = 0; // key for input
        for (NeuronContainer::iterator it = this->neurons->begin() + 1; it != this->neurons->end(); ++it) {
            TypeVector prepared = {input.at(i)};
            double result = (*it)->forward(prepared);

            output.push_back(result);
			(*this->a)(i + 1) = result;
            (*this->z)(i + 1) = result;

			i++;
        }
        return output;
    }
};

#endif /* INPUTLAYER_H */

