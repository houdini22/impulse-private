#ifndef PURELIN_H
#define PURELIN_H

#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/Neuron.h"
#include "OutputLayer.h"

class PurelinLayer : public OutputLayer {
public:

	PurelinLayer(int size, int prevSize) : OutputLayer(size, prevSize) {

	}

	double activation(double input) {
		return input;
	}

	double derivative(double input) {
		return 1.0;
	}
};

#endif /* OUTPUTLAYER_H */

