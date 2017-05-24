#include "stdafx.h"
#include "NetworkTrainer.h"
#include "Network.h"

NetworkTrainer::NetworkTrainer(Network * net) {
	this->network = net;
}

Network * NetworkTrainer::getNetwork() {
	return this->network;
}

NetworkTrainer * NetworkTrainer::setRegularization(double regularization) {
	this->regularization = regularization;
	return this;
}

NetworkTrainer * NetworkTrainer::setLearningIterations(int nb) {
	this->learningIterations = nb;
	return this;
}

CostResult NetworkTrainer::cost(DataSet dataSet) {
	Network * net = this->network;
	double sumErrors = 0.0;
	std::vector<MapSample> samples = dataSet.getSamples();

	// reset deltas
	for (LayerContainer::iterator it = net->getLayers()->begin() + 1; it != net->getLayers()->end() - 1; ++it) {
		for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
			Eigen::VectorXd * deltas = (*it2)->deltas;
			for (int i = 0; i < deltas->size(); i++) {
				(*deltas)(i) = 0;
			}
		}
	}

	for (int i = 0; i < samples.size(); i++) {
		Eigen::VectorXd predictions = net->forward(samples.at(i)["input"]);
		Eigen::VectorXd output = samples.at(i)["output"];
		net->backward(predictions, output);
		for (int j = 0; j < predictions.size(); j++) {
			sumErrors += pow(predictions(j) - output(j), 2.0);
			// sumErrors += ((output(j) * log(predictions(j))) + ((1 - output(j)) * log(1 - predictions(j))));
		}
	}

	double errorPenalty = 0.0;
	TypeVector resultGradient;
	LayerContainer * layers = net->getLayers();

	for (int i = 1; i < net->getSize() - 1; i++) {
		Eigen::MatrixXd penalty = layers->at(i)->backwardPenalty(dataSet.getSize(), this->regularization);
		errorPenalty += layers->at(i)->errorPenalty();
		Eigen::MatrixXd gradient = layers->at(i)->calculateGradient(dataSet.getSize(), penalty);
		Impulse::Math::Matrix::rollMatrixToVector(gradient, resultGradient);
	}

	double errorRegularization = (this->regularization * errorPenalty) / (2 * (double)dataSet.getSize());
	// double error = (-1.0 / (double) dataSet.getSize()) * sumErrors + errorRegularization;
	double error = (1.0 / (2.0 * (double)dataSet.getSize())) * sumErrors + errorRegularization;

	CostResult result;
	result.error = error;
	result.gradient = resultGradient;

	return result;
}

void NetworkTrainer::train(DataSet dataSet) {
	FmincgMy minimizer;
	Network * network = this->network;
	std::vector<double> theta = network->getRolledTheta();
	std::function<CostResult(std::vector<double>)> cf([this, &dataSet](std::vector<double> input)
	{
		this->setWeightsFromDenseVector(input);
		CostResult costResult = this->cost(dataSet);
		return costResult;
	});
	TypeVector optimized = minimizer.minimize(cf, theta, 100, true);
}

void NetworkTrainer::setWeightsFromDenseVector(std::vector<double> v) {
	this->network->setRolledTheta(v);
}