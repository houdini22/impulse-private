#ifndef FMINCG222_H
#define FMINCG222_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../Network/NetworkTrainer.h"

struct CostResult;

class FmincgMy {
public:
	FmincgMy() {};
	std::vector<double> minimize(std::function<CostResult(std::vector<double>)> costFunction, std::vector<double> theta, int length, bool verbose);
};

#endif
