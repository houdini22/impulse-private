#ifndef PROBLEM2_H
#define PROBLEM2_H

#include "../src/Data/DataSet.h"

#include "../src/cppoptlib/meta.h"
#include "../src/cppoptlib/problem.h"
#include "../src/cppoptlib/solver/conjugatedgradientdescentsolver.h"

class NetworkTrainer;
class DataSet;

using namespace cppoptlib;
using Eigen::VectorXd;

class Rosenbrock : public Problem<double> {
public:

	NetworkTrainer * trainer;
	DataSet dataSet;

	double value(const TVector &x);

	void gradient(const TVector &x, TVector &grad);

	void setNetworkTrainer(NetworkTrainer * trainer);

	void setDataSet(DataSet dataSet);
};

#endif

