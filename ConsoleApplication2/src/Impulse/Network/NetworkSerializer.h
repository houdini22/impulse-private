#include "stdafx.h"

#ifndef NETWORKSERIALIZER2_H
#define NETWORKSERIALIZER2_H

#include <fstream>
#include <string>
#include <iostream>

#include "Network.h"
#include "../../Vendor/json-2.1.1/src/json.hpp"

using json = nlohmann::json;

class NetworkSerializer {
protected:
	Network * network;
public:

	NetworkSerializer(Network * net) {
		this->network = net;
	}

	void toJSON(std::string path) {
		json result;

		std::vector<int> layersSize;
		for (LayerContainer::iterator it = this->network->getLayers()->begin(); it != this->network->getLayers()->end(); ++it) {
			auto * layer = (*it);
			layersSize.push_back(layer->getSize() - 1); // without bias neuron
		}
		result["layers"] = layersSize;

		TypeVector rolledTheta = this->network->getRolledTheta();
		result["neurons"] = rolledTheta;

		std::ofstream out(path);
		out << result.dump();
		out.close();
	}
};

#endif
