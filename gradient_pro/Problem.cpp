#include "stdafx.h"
#include "Problem.h"
#include <iostream>

#include "../src/Data/DataSet.h"
#include "../src/Network/NetworkTrainer.h"

#include "../src/cppoptlib/meta.h"
#include "../src/cppoptlib/problem.h"
#include "../src/cppoptlib/solver/conjugatedgradientdescentsolver.h"

using namespace cppoptlib;
using Eigen::VectorXd;

double Rosenbrock::value(const TVector &x) {
	std::vector<double> theta(x.data(), x.data() + x.size());
	this->trainer->getNetwork()->setRolledTheta(theta);
	CostResult result = this->trainer->cost(this->dataSet);
	std::cout << result.error << std::endl;
	return result.error;
}

void Rosenbrock::gradient(const TVector &x, TVector &grad) {
	std::vector<double> theta(x.data(), x.data() + x.size());
	this->trainer->getNetwork()->setRolledTheta(theta);
	CostResult result = this->trainer->cost(this->dataSet);
	for (int i = 0; i < grad.size(); i++) {
		grad[i] = result.gradient.at(i);
	}
}

void Rosenbrock::setNetworkTrainer(NetworkTrainer * trainer) {
	this->trainer = trainer;
}

void Rosenbrock::setDataSet(DataSet dataSet) {
	this->dataSet = dataSet;
}
