#include <iostream>

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
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "src/Impulse/NeuralNetwork/Network/NetworkBuilder.h"
#include "src/Impulse/NeuralNetwork/Network/NetworkTrainer.h"
#include "src/Impulse/NeuralNetwork/Data/DataSetManager.h"
#include "src/Impulse/NeuralNetwork/Network/NetworkSerializer.h"

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

Eigen::MatrixXd readMatrix(std::string path) {
    TypeMatrix matrix = readData(path);
    int rows = matrix.size();
    int cols = matrix.at(0).size();

    Eigen::MatrixXd result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = matrix.at(i).at(j);

    return result;
};

void TEST_Purelin() {
    Impulse::NeuralNetwork::Network::NetworkBuilder *builder = new Impulse::NeuralNetwork::Network::NetworkBuilder();
    // Network * network = builder->buildFromJSON("/home/hud/ml/purelin.json");
    builder->addInputLayer(1);
    builder->addPurelinLayer(4);
    builder->addPurelinLayer(1);
    builder->addOutputLayer();

    DataSetManager manager = DataSetManager();

    Eigen::MatrixXd input(4, 1);
    Eigen::MatrixXd output(4, 1);

    for (int i = 0; i < 4; i++) {
        input(i, 0) = i;
        output(i, 0) = i;
    }

    Impulse::NeuralNetwork::Network::Network *network = builder->getNetwork();

    DataSet dataSet = manager.createSet(input, output);
    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(network);

    trainer->setRegularization(0.0);
    trainer->setLearningIterations(200);
    std::cout << "Calculating cost." << std::endl;
    CostGradientResult result = trainer->cost(dataSet);
    std::cout << "Cost: " << result.getCost() << std::endl;

    std::cout << "Start training." << std::endl;
    trainer->train(dataSet);

    std::cout << network->forward(input.row(0)) << std::endl;
    std::cout << network->forward(input.row(1)) << std::endl;
    std::cout << network->forward(input.row(2)) << std::endl;
    std::cout << network->forward(input.row(3)) << std::endl;

    /*NetworkSerializer * serializer = new NetworkSerializer(network);
     std::string filename = "/home/hud/ml/purelin.json";
     serializer->toJSON(filename);*/
}

void TEST_my() {
    Impulse::NeuralNetwork::Network::NetworkBuilder *builder = new Impulse::NeuralNetwork::Network::NetworkBuilder();
    builder->addInputLayer(19200);
    builder->addPurelinLayer(300);
    builder->addPurelinLayer(100);
    builder->addPurelinLayer(4);
    builder->addOutputLayer();
    // Network * network = builder->buildFromJSON("/home/hud/ml/network.json");
    Impulse::NeuralNetwork::Network::Network *network = builder->getNetwork();

    DataSetManager manager = DataSetManager();

    Eigen::MatrixXd input(600, 19200);
    Eigen::MatrixXd output(600, 4);

    std::cout << "Loading dataset." << std::endl;

    int i = 0;
    int k = 0;

    boost::filesystem::path p("/var/www/mlgen/samples");
    boost::filesystem::directory_iterator end_itr;

    // cycle through the directory
    for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr) {
        if (boost::filesystem::is_regular_file(itr->path())) {
            std::ifstream fileStream(itr->path().string());
            json jsonFile;
            fileStream >> jsonFile;

            json x = jsonFile["x"];
            int j = 0;
            for (auto it = x.begin(); it != x.end(); ++it) {
                input(i, j) = it.value();
                j++;
            }

            json y = jsonFile["y"];
            j = 0;
            for (auto it = y.begin(); it != y.end(); ++it) {
                output(i, j) = it.value();
                j++;
            }

            if (i % 50 == 0) {
                std::cout << i << std::endl;
            }
            i++;
            if (i == 600) {
                break;
            }
        }
    }

    DataSet dataSet = manager.createSet(input, output);
    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(network);

    trainer->setRegularization(0.0);
    trainer->setLearningIterations(50);

    std::cout << "Calculating cost." << std::endl;
    CostGradientResult result = trainer->cost(dataSet);
    std::cout << "Cost: " << result.getCost() << std::endl;

    for (int training = 0; training < 10; training++) {
        std::cout << "Start training." << std::endl;
        trainer->train(dataSet);
        std::cout << "Calculating cost." << std::endl;
        CostGradientResult result = trainer->cost(dataSet);
        std::cout << "Cost: " << result.getCost() << std::endl;

        Impulse::NeuralNetwork::Network::NetworkSerializer *serializer = new Impulse::NeuralNetwork::Network::NetworkSerializer(network);
        std::string filename = "/home/hud/ml/network.json";
        filename.append(std::to_string(training));
        serializer->toJSON(filename);
    }

}

void TEST_LogisticRegression() {
    Impulse::NeuralNetwork::Network::NetworkBuilder builder = Impulse::NeuralNetwork::Network::NetworkBuilder();
    builder.addInputLayer(400);
    builder.addLogisticLayer(20);
    builder.addLogisticLayer(10);
    builder.addOutputLayer();

    Impulse::NeuralNetwork::Network::Network *net = builder.getNetwork();

    // load input
    Eigen::MatrixXd input = readMatrix(
            "/home/hud/CLionProjects/impulse/data/ex4data1_x.txt");

    // load output
    Eigen::MatrixXd output = readMatrix(
            "/home/hud/CLionProjects/impulse/data/ex4data1_y.txt");

    DataSetManager manager = DataSetManager();
    DataSet dataSet = manager.createSet(input, output);

    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(net);
    trainer->setLearningIterations(400);

    std::cout << "Calculating cost." << std::endl;

    CostGradientResult result = trainer->cost(dataSet);
    std::cout << "Cost: " << result.error << std::endl;

    std::cout << net->forward(input.row(0)) << std::endl;

    std::cout << "Start training." << std::endl;
    trainer->train(dataSet);

    CostGradientResult result2 = trainer->cost(dataSet);
    std::cout << "Cost: " << result2.error << std::endl;

    std::cout << net->forward(input.row(0)) << std::endl;

    Impulse::NeuralNetwork::Network::NetworkSerializer * serializer = new Impulse::NeuralNetwork::Network::NetworkSerializer(net);
    std::string filename = "/home/hud/CLionProjects/impulse-new/cmake-build-release/logistic.json";
    serializer->toJSON(filename);

    std::cout << "Saved." << std::endl;
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
}

void TEST_LogisticLoad() {
    Impulse::NeuralNetwork::Network::NetworkBuilder * builder = new Impulse::NeuralNetwork::Network::NetworkBuilder();
    Impulse::NeuralNetwork::Network::Network * net = builder->buildFromJSON("/home/hud/CLionProjects/impulse-new/cmake-build-release/logistic.json");

    // load input
    Eigen::MatrixXd input = readMatrix(
            "/home/hud/CLionProjects/impulse/data/ex4data1_x.txt");

    // load output
    Eigen::MatrixXd output = readMatrix(
            "/home/hud/CLionProjects/impulse/data/ex4data1_y.txt");

    DataSetManager manager = DataSetManager();
    DataSet dataSet = manager.createSet(input, output);

    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(net);

    CostGradientResult result2 = trainer->cost(dataSet);
    std::cout << "Cost: " << result2.error << std::endl;

    std::cout << net->forward(input.row(0)) << std::endl;
}

int main() {
    //TEST_my();
    //TEST_Purelin();
    //TEST_LogisticRegression();
    TEST_LogisticLoad();
    getchar();
    return 0;
}

