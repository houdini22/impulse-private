#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "../Neuron/AbstractNeuron.h"
#include "../Neuron/BiasNeuron.h"
#include "../Neuron/Neuron.h"
#include "HiddenLayer.h"

class OutputLayer : public HiddenLayer {
public:

    OutputLayer(int size, int prevSize) : HiddenLayer(size, prevSize) {
		this->a = new Eigen::VectorXd(prevSize);
    }

    TypeVector forward(TypeVector input) {
        TypeVector result = HiddenLayer::forward(input);
        // remove first bias output
        result.erase(result.begin());
        return result;
    }
};

#endif /* OUTPUTLAYER_H */

