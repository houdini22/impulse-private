#ifndef NETWORKTRAINER_H
#define NETWORKTRAINER_H

#include <functional>
#include <string>
#include <stdio.h>
#include <memory>
#include <cstring>

#include "Network.h"
#include "../Data/DataSet.h"
#include "../Math/Matrix.h"
#include "../Math/FmincgMy.h"

struct CostResult {
	double error;
	TypeVector gradient;
};

class NetworkTrainer {
protected:
    Network * network;
    double regularization = 0.0;
    int learningIterations = 1;
public:
	NetworkTrainer(Network * net);
	Network * getNetwork();
	NetworkTrainer * setRegularization(double regularization);
	NetworkTrainer * setLearningIterations(int nb);
	CostResult cost(DataSet dataSet);
	void train(DataSet dataSet);
	void setWeightsFromDenseVector(std::vector<double> v);
};

#endif /* NETWORKTRAINER_H */

