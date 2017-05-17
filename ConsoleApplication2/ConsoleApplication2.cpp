#include "stdafx.h"
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ios>
#include "src/Network/NetworkBuilder.h"
#include "src/Network/NetworkTrainer.h"
#include "src/Data/DataSetManager.h"
#include "src/Data/DataSetManager.h"

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


int main()
{
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
	TypeMatrix input = readData("data/ex4data1_x.txt");

	// load output
	TypeMatrix output = readData("data/ex4data1_y.txt");

	/*
	for (int i = 0; i < input.size(); i++) {
	net->forward(input.at(i));
	}
	*/

	std::cout << input.size() << std::endl;

	TypeVector netOutput = net->forward(input.at(0));
	for (int i = 0; i < netOutput.size(); i++) {
		std::cout << "Output " << i << ": " << netOutput.at(i) << std::endl;
	}

	DataSetManager manager = DataSetManager();
	DataSet dataSet = manager.createSet(input, output);

	NetworkTrainer * trainer = new NetworkTrainer(net);

	trainer->setRegularization(0.0);
	trainer->setLearningIterations(50);

	CostResult result = trainer->cost(dataSet);
	std::cout << "Cost: " << result.error << std::endl;

	std::cout << "Start training." << std::endl;
	trainer->train(dataSet);
	CostResult result2 = trainer->cost(dataSet);
	std::cout << "Cost: " << result2.error << std::endl;

	getchar();

    return 0;
}

