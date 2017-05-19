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

#include "../../Vendor/tjungblut-math-cpp/CostGradientTuple.h"
#include "../../Vendor/tjungblut-math-cpp/DenseVector.h"
#include "../../Vendor/tjungblut-math-cpp/Fmincg.h"

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

    NetworkTrainer(Network * net) {
        this->network = net;
    }

	Network * getNetwork() {
		return this->network;
	}

    NetworkTrainer * setRegularization(double regularization) {
        this->regularization = regularization;
        return this;
    }

    NetworkTrainer * setLearningIterations(int nb) {
        this->learningIterations = nb;
        return this;
    }

    CostResult cost(DataSet & dataSet) {
        Network * net = this->network;
        double sumErrors = 0.0;
        std::vector<MapSample> samples = dataSet.getSamples();

		// reset deltas
		for (LayerContainer::iterator it = net->getLayers()->begin() + 1; it != net->getLayers()->end(); ++it) {
			for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
				Eigen::VectorXd * deltas = (*it2)->deltas;
				for (int i = 0; i < deltas->size(); i++) {
					(*deltas)(i) = 0;
				}
			}
		}

        for (int i = 0; i < samples.size(); i++) {
            TypeVector predictions = net->forward(samples.at(i)["input"]);
            TypeVector output = samples.at(i)["output"];
            net->backward(predictions, output);
            for (int j = 0; j < predictions.size(); j++) {
                sumErrors += ((output.at(j) * log(predictions.at(j))) + ((1 - output.at(j)) * log(1 - predictions.at(j))));
            }
        }

        double errorPenalty = 0.0;
        TypeVector resultGradient;
        LayerContainer * layers = net->getLayers();

        for (int i = 1; i < net->getSize(); i++) {
            TypeMatrix penalty = layers->at(i)->backwardPenalty(dataSet.getSize(), this->regularization);
            errorPenalty += layers->at(i)->errorPenalty();
            TypeMatrix gradient = layers->at(i)->calculateGradient(dataSet.getSize(), penalty);
            Impulse::Math::Matrix::rollMatrixToVector(gradient, resultGradient);
        }

        double errorRegularization = (this->regularization * errorPenalty) / (2 * (double) dataSet.getSize());
        double error = (-1.0 / (double) dataSet.getSize()) * sumErrors + errorRegularization;

        CostResult result;
        result.error = error;
        result.gradient = resultGradient;

		for (LayerContainer::iterator it = net->getLayers()->begin() + 1; it != net->getLayers()->end(); ++it) {
			for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
				Eigen::VectorXd * deltas = (*it2)->deltas;
				for (int i = 0; i < 1; i++) {
					std::cout << (*deltas)(i) << std::endl;
					break;
				}
				break;
			}
			break;
		}

        return result;
    }
   
	void train(DataSet dataSet) {
		tjmath::Fmincg<double> minimizer;

		Network * network = this->network;

		std::vector<double> rolledTheta = network->getRolledTheta();
		tjmath::DenseVector<double> theta(rolledTheta.size(), rolledTheta);

		std::function<tjmath::CostGradientTuple<double>(tjmath::DenseVector<double>)> cf([this, &dataSet](tjmath::DenseVector<double> input)
		{
			this->setWeightsFromDenseVector(input);
			CostResult costResult = this->cost(dataSet);
			tjmath::DenseVector<double> gradient(costResult.gradient.size(), costResult.gradient);

			return tjmath::CostGradientTuple<double>(costResult.error, gradient);
		});
		tjmath::DenseVector<double> optimizedParameters = minimizer.minimize(cf, theta, this->learningIterations, true);

		this->setWeightsFromDenseVector(optimizedParameters);
	}

	void setWeightsFromDenseVector(tjmath::DenseVector<double> v) {
		std::vector<double> v2;
		for (int i = 0; i < v.getDimension(); i++) {
			v2.push_back(v.get(i));
		}
		this->network->setRolledTheta(v2);
	}
};

#endif /* NETWORKTRAINER_H */

