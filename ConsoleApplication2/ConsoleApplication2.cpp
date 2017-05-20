#include "stdafx.h"
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ios>
#include <ctime>
#include <experimental/filesystem>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace fs = std::experimental::filesystem::v1;

#include "src/Impulse/Network/NetworkBuilder.h"
#include "src/Impulse/Network/NetworkTrainer.h"
#include "src/Impulse/Data/DataSetManager.h"
#include "src/Impulse/Data/DataSetManager.h"
#include "src/Impulse/Network/NetworkSerializer.h"

double strToDouble(std::string str) {
	std::istringstream os(str);
	double d;
	os >> d;
	return d;
}

TypeMatrix readData(std::string path) {
	std::ifstream file(path);
	std::string line;

	TypeMatrix result;

	while (std::getline(file, line)) {
		TypeVector lineData;
		std::istringstream iss(line);
		std::string token;
		while (std::getline(iss, token, ' ')) {
			lineData.push_back(strToDouble(token));
		}
		result.push_back(lineData);
	}

	return result;
}

Eigen::MatrixXd readMatrix(std::string path)
{
	TypeMatrix matrix = readData(path);
	int rows = matrix.size();
	int cols = matrix.at(0).size();

	Eigen::MatrixXd result(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			result(i, j) = matrix.at(i).at(j);

	return result;
};


int main()
{
	Eigen::initParallel();
	// omp_set_num_threads(4);
	// Eigen::setNbThreads(4);

	/*MOJE
	NetworkBuilder * builder = new NetworkBuilder();

	// Network * net = builder.buildFromJSON("e:/network.json");

	clock_t buildStart = clock();

	builder->addInputLayer(38400);
	builder->addHiddenLayer(300);
	builder->addOutputLayer(2);
	
	DataSetManager manager = DataSetManager();

	Eigen::MatrixXd input(500, 38400);
	Eigen::MatrixXd output(500, 2);

	std::cout << "Loading dataset." << std::endl;

	std::string path = "E:\\impulse\\gen2\\samples";
	int i = 0;
	for (auto & p : fs::directory_iterator(path)) {
		if (i == 500) break;

		if (i % 50 == 0) {
			std::cout << i << std::endl;
		}

		std::stringstream path;
		path << p;

		std::ifstream fileStream(path.str());
		json jsonFile;
		fileStream >> jsonFile;

		json x = jsonFile["x"];
		int j = 0;
		for (auto it = x.begin(); it != x.end(); ++it) {
			input(i, j) = it.value();
			j++;
		}
		
		double outputX = jsonFile["y"]["x"];
		double outputY = jsonFile["y"]["y"];

		output(i, 0) = outputX;
		output(i, 1) = outputY;

		i++;
	}

	// std::cout << "Restoring network state." << std::endl;

	Network * network = builder->getNetwork();
	// Network * network = builder->buildFromJSON("e:/network.json");
	DataSet dataSet = manager.createSet(input, output);
	NetworkTrainer * trainer = new NetworkTrainer(network);

	trainer->setRegularization(0.0);
	trainer->setLearningIterations(50);

	// std::cout << "Calculating cost." << std::endl;
	// CostResult result = trainer->cost(dataSet);
	// std::cout << "Cost: " << result.error << std::endl;

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	NetworkSerializer * serializer = new NetworkSerializer(network);
	serializer->toJSON("e:/network.json");
	*/

	
	NetworkBuilder builder = NetworkBuilder();

	builder.addInputLayer(400);
	builder.addHiddenLayer(25);
	builder.addOutputLayer(10);
	Network * net = builder.getNetwork();

	// std::cout << net->getSize() << std::endl; // check network size

	// load weights
	TypeMatrix w1 = readData("data/ex4weights_1.txt");
	TypeMatrix w2 = readData("data/ex4weights_2.txt");

	// set weights
	int i = 0;
	for (LayerContainer::iterator it = net->getLayers()->begin(); it != net->getLayers()->end(); ++it) {
		if (i > 0) {
			int k = 0;
			for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
				Eigen::VectorXd * weights = (*it2)->weights;
				for (int j = 0; j < weights->size(); j++) {
					if (i == 1) {
						(*weights)(j) = w1.at(k).at(j);
					}
					else {
						(*weights)(j) = w2.at(k).at(j);
					}
				}
				k++;
			}
		}
		i++;
	}

	// debug weights
	// std::cout << "[1][1][0]" << net->getLayers()->at(1)->getNeurons()->at(1)->weights->at(0) << std::endl;
	// std::cout << "[1][1][400]" << net->getLayers()->at(1)->getNeurons()->at(1)->weights->at(400) << std::endl;
	// std::cout << "[1][25][0]" << net->getLayers()->at(1)->getNeurons()->at(25)->weights->at(0) << std::endl;
	// std::cout << "[1][25][400]" << net->getLayers()->at(1)->getNeurons()->at(25)->weights->at(400) << std::endl;

	// std::cout << "[2][1][0]" << net->getLayers()->at(2)->getNeurons()->at(1)->weights->at(0) << std::endl;
	// std::cout << "[2][1][25]" << net->getLayers()->at(2)->getNeurons()->at(1)->weights->at(25) << std::endl;
	// std::cout << "[2][10][0]" << net->getLayers()->at(2)->getNeurons()->at(10)->weights->at(0) << std::endl;
	// std::cout << "[2][10][25]" << net->getLayers()->at(2)->getNeurons()->at(10)->weights->at(25) << std::endl;

	// load input
	Eigen::MatrixXd input = readMatrix("data/ex4data1_x.txt");

	// load output
	Eigen::MatrixXd output = readMatrix("data/ex4data1_y.txt");

	DataSetManager manager = DataSetManager();
	DataSet dataSet = manager.createSet(input, output);

	NetworkTrainer * trainer = new NetworkTrainer(net);
	trainer->setLearningIterations(100);

	std::cout << "Calculating cost." << std::endl;

	CostResult result = trainer->cost(dataSet);
	std::cout << "Cost: " << result.error << std::endl;

	std::cout << net->forward(input.row(0)) << std::endl;
	
	return 0;

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);

	
	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);

	result = trainer->cost(dataSet);
	std::cout << "Cost: " << result.error << std::endl;
	
	/*
	5000
	Output 0: 0.000112662
	Output 1: 0.00174128
	Output 2: 0.00252697
	Output 3: 1.84032e-05
	Output 4: 0.00936264
	Output 5: 0.0039927
	Output 6: 0.00551518
	Output 7: 0.000401468
	Output 8: 0.00648072
	Output 9: 0.995734
	0.309359
	Cost: 0.287629
	*/

	/*
	EIGEN
	Eigen::VectorXd vec(10);
	for (int i = 0; i < vec.size(); ++i) {
	vec[i] = i;
	}
	std::cout << vec << '\n';

	return 0;
	NetworkBuilder * builder = new NetworkBuilder();

	builder->addInputLayer(38400);
	builder->addHiddenLayer(300);
	builder->addOutputLayer(2);

	DataSetManager manager = DataSetManager();

	TypeMatrix input;
	TypeMatrix output;

	std::cout << "Loading dataset." << std::endl;

	std::string path = "E:\\impulse\\gen2\\samples";
	int i = 0;
	for (auto & p : fs::directory_iterator(path)) {
	if (i > 500) break;

	if (i % 100 == 0) {
	std::cout << i << std::endl;
	}

	std::stringstream path;
	path << p;

	std::ifstream fileStream(path.str());
	json jsonFile;
	fileStream >> jsonFile;

	TypeVector inputRow;
	json x = jsonFile["x"];
	for (auto it = x.begin(); it != x.end(); ++it) {
	inputRow.push_back(it.value());
	}
	input.push_back(inputRow);

	double outputX = jsonFile["y"]["x"];
	double outputY = jsonFile["y"]["y"];

	TypeVector outputRow;
	outputRow.push_back(outputX);
	outputRow.push_back(outputY);

	output.push_back(outputRow);

	i++;
	}

	Network * network = builder->getNetwork();
	std::cout << "Builing net." << std::endl;
	DataSet dataSet = manager.createSet(input, output);

	NetworkTrainer * trainer = new NetworkTrainer(network);

	trainer->setRegularization(0.0);
	trainer->setLearningIterations(3);

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	NetworkSerializer * serializer = new NetworkSerializer(network);
	serializer->toJSON("e:/network.json");
	*/

	// TypeVector netOut = network->forward(input.at(0));

	// std::cout << netOut.at(0) << " " << netOut.at(1) << std::endl;
	// std::cout << output.at(0).at(0) << " " << output.at(0).at(1) << std::endl;
	/*
	for (int i = 0; i < 10; i++) {
	TypeVector netOut = network->forward(input.at(i * 15));
	std::stringstream result;
	std::copy(netOut.begin(), netOut.end(), std::ostream_iterator<double>(result, " "));

	std::stringstream result2;
	std::copy(output.at(i * 15).begin(), output.at(i * 15).end(), std::ostream_iterator<double>(result2, " "));

	std::cout << result.str() << " " << result2.str() << std::endl;
	}

	NetworkTrainer * trainer = new NetworkTrainer(network);

	trainer->setRegularization(0.0);

	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	trainer->setLearningIterations(3);

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	NetworkSerializer * serializer = new NetworkSerializer(network);
	serializer->toJSON("e:/network.json");
	*/


	/*
	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;
	*/

	/*
	for (int i = 0; i < 10; i++) {
	clock_t begin = clock();
	TypeVector netOutput = network->forward(input.at(i));
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	for (int i = 0; i < netOutput.size(); i++) {
	std::cout << "Output " << i << ": " << netOutput.at(i) << std::endl;
	}
	std::cout << elapsed_secs << std::endl;
	}
	*/

	/*
	clock_t buildEnd = clock();
	std::cout << "Build: " << double(buildEnd - buildStart) / CLOCKS_PER_SEC << std::endl;


	Network * net = builder->getNetwork();

	TypeVector testInput;
	for (int i = 0; i < 38400; i++) {
	testInput.push_back(1);
	}

	for (int i = 0; i < 10; i++) {
	clock_t begin = clock();
	TypeVector netOutput = net->forward(testInput);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << elapsed_secs << std::endl;
	}
	*/

	/*
	clock_t begin = clock();
	TypeVector netOutput = net->forward(testInput);
	clock_t end = clock();

	for (int i = 0; i < netOutput.size(); i++) {
	std::cout << "Output " << i << ": " << netOutput.at(i) << std::endl;
	}

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << elapsed_secs << std::endl;
	*/

	// std::cout << net->getSize() << std::endl; // check network size

	// load weights
	// TypeMatrix w1 = readData("data/ex4weights_1.txt");
	// TypeMatrix w2 = readData("data/ex4weights_2.txt");

	// set weights
	/*
	int i = 0;
	for (LayerContainer::iterator it = net->getLayers()->begin(); it != net->getLayers()->end(); ++it) {
	if (i > 0) {
	int k = 0;
	for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
	TypeVector * weights = (*it2)->weights;
	for (int j = 0; j < weights->size(); j++) {
	if (i == 1) {
	weights->at(j) = w1.at(k).at(j);
	}
	else {
	weights->at(j) = w2.at(k).at(j);
	}
	}
	k++;
	}
	}
	i++;
	}
	*/

	// debug weights
	// std::cout << "[1][1][0]" << net->getLayers()->at(1)->getNeurons()->at(1)->weights->at(0) << std::endl;
	// std::cout << "[1][1][400]" << net->getLayers()->at(1)->getNeurons()->at(1)->weights->at(400) << std::endl;
	// std::cout << "[1][25][0]" << net->getLayers()->at(1)->getNeurons()->at(25)->weights->at(0) << std::endl;
	// std::cout << "[1][25][400]" << net->getLayers()->at(1)->getNeurons()->at(25)->weights->at(400) << std::endl;

	// std::cout << "[2][1][0]" << net->getLayers()->at(2)->getNeurons()->at(1)->weights->at(0) << std::endl;
	// std::cout << "[2][1][25]" << net->getLayers()->at(2)->getNeurons()->at(1)->weights->at(25) << std::endl;
	// std::cout << "[2][10][0]" << net->getLayers()->at(2)->getNeurons()->at(10)->weights->at(0) << std::endl;
	// std::cout << "[2][10][25]" << net->getLayers()->at(2)->getNeurons()->at(10)->weights->at(25) << std::endl;

	// load input
	// TypeMatrix input = readData("data/ex4data1_x.txt");

	// load output
	// TypeMatrix output = readData("data/ex4data1_y.txt");

	/*
	for (int i = 0; i < input.size(); i++) {
	net->forward(input.at(i));
	}
	*/

	// std::cout << input.size() << std::endl;

	/*
	TypeVector netOutput = net->forward(input.at(0));
	for (int i = 0; i < netOutput.size(); i++) {
	std::cout << "Output " << i << ": " << netOutput.at(i) << std::endl;
	}

	DataSetManager manager = DataSetManager();
	DataSet dataSet = manager.createSet(input, output);

	NetworkTrainer * trainer = new NetworkTrainer(net);

	trainer->setRegularization(0.0);
	trainer->setLearningIterations(150);

	CostResult result = trainer->cost(dataSet);
	std::cout << "Cost: " << result.error << std::endl;

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	TypeVector netOutput2 = net->forward(input.at(0));
	for (int i = 0; i < netOutput2.size(); i++) {
	std::cout << "Output " << i << ": " << netOutput2.at(i) << std::endl;
	}
	*/

	/*
	NetworkSerializer * serializer = new NetworkSerializer(net);
	serializer->toJSON("e:/network.json");
	*/

	getchar();

	return 0;
}

