#ifndef NETWORKBUILDER_H
#define NETWORKBUILDER_H

#include "Network.h"
#include "../Layer/InputLayer.h"
#include "../Layer/HiddenLayer.h"
#include "../Layer/OutputLayer.h"
#include "../../Vendor/json-2.1.1/src/json.hpp"

using json = nlohmann::json;

class NetworkBuilder {
protected:
    Network * network;
    int prevSize;
public:

    NetworkBuilder() {
        this->network = new Network();
    }

    NetworkBuilder * addInputLayer(int size) {
        InputLayer * layer = new InputLayer(size);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder * addHiddenLayer(int size) {
        HiddenLayer * layer = new HiddenLayer(size, this->prevSize);
        this->network->addLayer(layer);
        this->prevSize = layer->getSize();
        return this;
    }

    NetworkBuilder * addOutputLayer(int size) {
        OutputLayer * layer = new OutputLayer(size, this->prevSize);
        this->network->addLayer(layer);
        return this;
    }

    Network * getNetwork() {
        return this->network;
    }

	void clear() {
		delete this->network;
	}

	Network * buildFromJSON(std::string path) {
		std::ifstream fileStream(path);
		json jsonFile;
		fileStream >> jsonFile;
		fileStream.close();

		json savedLayers = jsonFile["layers"];
		int i = 0;
		for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it)
		{
			if (i == 0) {
				this->addInputLayer(it.value());
			}
			else if (i == savedLayers.size() - 1) {
				this->addOutputLayer(it.value());
			}
			else {
				this->addHiddenLayer(it.value());
			}
			i++;
		}

		TypeVector theta;
		json savedTheta = jsonFile["neurons"];
		for (auto it = savedTheta.begin(); it != savedTheta.end(); ++it) {
			theta.push_back(it.value());
		}

		this->network->setRolledTheta(theta);

		return this->network;
	}
};

#endif /* NETWORKBUILDER_H */

