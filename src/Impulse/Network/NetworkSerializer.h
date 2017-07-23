#ifndef NETWORKSERIALIZER2_H
#define NETWORKSERIALIZER2_H

#include <fstream>
#include <string>
#include <iostream>

#include "Network.h"
#include "../../Vendor/json.hpp"

using json = nlohmann::json;

class NetworkSerializer {
protected:
    Network *network;
public:

    NetworkSerializer(Network *net) {
        this->network = net;
    }

    void toJSON(std::string path) {
        json result;

        std::vector<int> layersSize;
        for (LayerContainer::iterator it = this->network->getLayers()->begin();
             it != this->network->getLayers()->end() - 1; ++it) {
            auto *layer = (*it);
            layersSize.push_back(layer->getSize() - 1); // without bias neuron
        }
        result["layers"] = layersSize;

        Eigen::VectorXd rolledTheta = this->network->getRolledTheta();
        result["neurons"] = std::vector<double>(rolledTheta.data(), rolledTheta.data() + rolledTheta.size());

        std::ofstream out(path);
        out << result.dump();
        out.close();
    }
};

#endif
