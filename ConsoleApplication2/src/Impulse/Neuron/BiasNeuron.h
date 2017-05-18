#ifndef BIASNEURON_H
#define BIASNEURON_H

#include "AbstractNeuron.h"

class BiasNeuron : public AbstractNeuron {
public:

    BiasNeuron() : AbstractNeuron(1) {
    }

    double forward(TypeVector input) {
        return 1.0;
    }
};

#endif /* BIASNEURON_H */

