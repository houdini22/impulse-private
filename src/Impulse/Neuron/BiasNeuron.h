#ifndef BIASNEURON_H
#define BIASNEURON_H

#include "AbstractNeuron.h"

class BiasNeuron : public AbstractNeuron {
public:

    BiasNeuron() : AbstractNeuron(1) {
    }

    double forward(Eigen::VectorXd input) {
        return (*this->weights)(0);
    }
};

#endif /* BIASNEURON_H */

