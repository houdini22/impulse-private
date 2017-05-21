#ifndef PURELINLAYER_H
#define PURELINLAYER_H

#include "HiddenLayer.h"

class PurelinLayer : public HiddenLayer {
public:

	PurelinLayer(int size, int prevSize) : HiddenLayer(size, prevSize) {

	}

	double activation(double input) {
		return input;
	}

	double derivative(double input) {
		return 1.0;
	}
};

#endif /* OUTPUTLAYER_H */

