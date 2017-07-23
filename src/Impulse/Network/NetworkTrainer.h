#pragma once

#include <functional>
#include <string>
#include <stdio.h>
#include <memory>
#include <cstring>

#include "Network.h"
#include "../Data/DataSet.h"
#include "../Math/Matrix.h"
#include "../Math/Minimizer/Fmincg.h"

struct CostGradientResult {
	double error;
	Eigen::VectorXd gradient;
	double getCost() {
		return this->error;
	}
	Eigen::VectorXd getGradient() {
		return this->gradient;
	}
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

	CostGradientResult cost(DataSet & dataSet);

	void train(DataSet dataSet);
};
