#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "AbstractLayer.h"

class OutputLayer : public AbstractLayer {
public:

    OutputLayer() : AbstractLayer(0, 0) {
    
	}

	Eigen::VectorXd forward(Eigen::VectorXd input) {
		Eigen::VectorXd newResult(input.size() - 1);
		for (int i = 1; i < input.size(); i++) {
			newResult(i - 1) = input(i);
		}
		return newResult;
    }
};

#endif /* OUTPUTLAYER_H */

