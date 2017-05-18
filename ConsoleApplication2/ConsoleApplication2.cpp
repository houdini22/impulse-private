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


int main()
{
	NetworkBuilder * builder = new NetworkBuilder();
	// Network * net = builder.buildFromJSON("e:/network.json");

	builder->addInputLayer(153600);
	builder->addHiddenLayer(300);
	builder->addOutputLayer(9);
	Network * net = builder->getNetwork();

	TypeVector testInput;
	for (int i = 0; i < 153600; i++) {
		testInput.push_back(1);
	}
	
	for (int i = 0; i < 10; i++) {
		clock_t begin = clock();
		TypeVector netOutput = net->forward(testInput);
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << elapsed_secs << std::endl;
	}

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

