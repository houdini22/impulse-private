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

	Eigen::VectorXd forward(Eigen::VectorXd input) {
		Eigen::VectorXd result = HiddenLayer::forward(input);
		Eigen::VectorXd newResult(result.size() - 1);
		for (int i = 1; i < result.size(); i++) {
			newResult(i - 1) = result(i);
		}
		return newResult;
    }
};

#endif /* OUTPUTLAYER_H */

