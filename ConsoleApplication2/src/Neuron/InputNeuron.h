#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include "AbstractNeuron.h"

class InputNeuron : public AbstractNeuron {
public:

    InputNeuron() : AbstractNeuron(1) {
    }

    double forward(TypeVector input) {
        return input[0];
    }
};

#endif /* INPUTNEURON_H */

