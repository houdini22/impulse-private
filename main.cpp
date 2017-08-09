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

#include "src/Impulse/NeuralNetwork/Builder/ClassificationBuilder.h"
#include "src/Impulse/NeuralNetwork/Builder/RegressionBuilder.h"
#include "src/Impulse/NeuralNetwork/Network/NetworkTrainer.h"
#include "src/Impulse/NeuralNetwork/Network/NetworkSerializer.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

void TEST_Regression() {
    Impulse::NeuralNetwork::Builder::RegressionBuilder builder;
    builder.createInputLayer(1);
    builder.createHiddenLayer(4);
    builder.createHiddenLayer(1);
    builder.addOutputLayer();

    Impulse::SlicedDataset data;

    for (double i = 0.0; i < 4.0; i = i + 1.0) {
        data.input.addSample(Impulse::DatasetSample({i}));
        data.output.addSample(Impulse::DatasetSample({i}));
    }

    Impulse::NeuralNetwork::Network::Network *network = builder.getNetwork();

    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(
            network);

    trainer->setRegularization(0.0);
    trainer->setLearningIterations(200);
    std::cout << "Calculating cost." << std::endl;
    Impulse::NeuralNetwork::Network::CostGradientResult result = trainer->cost(&data);
    std::cout << "Cost: " << result.getCost() << std::endl;

    std::cout << "Start training." << std::endl;
    trainer->train(&data);

    std::cout << network->forward(data.inputVector.row(0)) << std::endl;
    std::cout << network->forward(data.inputVector.row(1)) << std::endl;
    std::cout << network->forward(data.inputVector.row(2)) << std::endl;
    std::cout << network->forward(data.inputVector.row(3)) << std::endl;

    Impulse::NeuralNetwork::Network::NetworkSerializer *serializer = new Impulse::NeuralNetwork::Network::NetworkSerializer(network);
    std::string filename = "/home/hud/ml/purelin.json";
    serializer->toJSON(filename);
    std::cout << "Saved." << std::endl;
}

/*void TEST_my() {
    Impulse::NeuralNetwork::Network::NetworkBuilder *builder = new Impulse::NeuralNetwork::Network::NetworkBuilder();
    builder->createInputLayer(19200);
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

}*/

void TEST_Classification() {
    // build network

    Impulse::NeuralNetwork::Builder::ClassificationBuilder builder;
    builder.createInputLayer(400);
    builder.createHiddenLayer(20);
    builder.createHiddenLayer(10);
    builder.addOutputLayer();

    Impulse::NeuralNetwork::Network::Network *net = builder.getNetwork();
    // end build network

    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;
    // end create dataset

    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(net);
    trainer->setLearningIterations(400);

    std::cout << "Calculating cost." << std::endl;

    Impulse::NeuralNetwork::Network::CostGradientResult result = trainer->cost(&dataset);
    std::cout << "Cost: " << result.error << std::endl;

    std::cout << net->forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    std::cout << "Start training." << std::endl;
    trainer->train(&dataset);

    Impulse::NeuralNetwork::Network::CostGradientResult result2 = trainer->cost(&dataset);
    std::cout << "Cost: " << result2.error << std::endl;

    std::cout << net->forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Network::NetworkSerializer *serializer = new Impulse::NeuralNetwork::Network::NetworkSerializer(
            net);
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

void TEST_ClassificationLoad() {
    Impulse::NeuralNetwork::Builder::ClassificationBuilder * builder = new Impulse::NeuralNetwork::Builder::ClassificationBuilder();
    Impulse::NeuralNetwork::Network::Network * net = builder->buildFromJSON("/home/hud/CLionProjects/impulse-new/cmake-build-release/logistic.json");
    std::cout << "Loaded." << std::endl;

    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;
    // end create dataset

    Impulse::NeuralNetwork::Network::NetworkTrainer *trainer = new Impulse::NeuralNetwork::Network::NetworkTrainer(net);

    std::cout << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Network::CostGradientResult result = trainer->cost(&dataset);
    std::cout << "Cost: " << result.error << std::endl;
}

int main() {
    //TEST_my();
    //TEST_Regression();
    //TEST_Classification();
    TEST_ClassificationLoad();
    getchar();
    return 0;
}

