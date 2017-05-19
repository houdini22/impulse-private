#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "../Layer/AbstractLayer.h"

typedef std::vector<AbstractLayer*> LayerContainer;

class Network {
protected:

    int size = 0;
    LayerContainer * layers;

public:

    Network() {
        this->layers = new LayerContainer;
    }

    ~Network() {
        for (auto i = this->layers->begin(); i != this->layers->end(); i++) {
            delete *i;
        }
        this->layers->clear();
        delete this->layers;
    }

    void addLayer(AbstractLayer * layer) {
        this->size++;
        this->layers->push_back(layer);
    }

    int getSize() {
        return this->size;
    }

    LayerContainer * getLayers() {
        return this->layers;
    }

    TypeVector forward(TypeVector input) {
        TypeVector output = input;
        for (LayerContainer::iterator it = this->layers->begin(); it != this->layers->end(); ++it) {
            output = (*it)->forward(output);
        }
        return output;
    }

    void backward(TypeVector predictions, TypeVector output) {
		Eigen::VectorXd sigma(predictions.size());
        for (int i = 0; i < predictions.size(); i++) {
            sigma(i) = predictions.at(i) - output.at(i);
        }

        for (int i = this->size - 1; i > 0; i--) {
            auto * layer = this->layers->at(i);
            auto * prevLayer = this->layers->at(i - 1);

            layer->calculateDeltas(prevLayer->getA(), sigma);
            sigma = prevLayer->backward(sigma, layer);
        }
    }

	std::vector<double> getRolledTheta() {
		std::vector<double> result;
		for (auto layerIterator = this->layers->begin() + 1; layerIterator != this->layers->end(); layerIterator++) {
			for (NeuronContainer::iterator neuronIterator = (*layerIterator)->getNeurons()->begin() + 1; neuronIterator != (*layerIterator)->getNeurons()->end(); ++neuronIterator) {
				Eigen::VectorXd * weights = (*neuronIterator)->weights;
				for (int j = 0; j < weights->size(); j++) {
					result.push_back((*weights)(j));
				}
			}
		}
		return result;
	}

	void setRolledTheta(std::vector<double> rolledTheta) {
		int i = 0;
		for (LayerContainer::iterator it = this->getLayers()->begin() + 1; it != this->getLayers()->end(); ++it) {
			for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1; it2 != (*it)->getNeurons()->end(); ++it2) {
				Eigen::VectorXd * weights = (*it2)->weights;
				for (int j = 0; j < weights->size(); j++) {
					(*weights)(j) = rolledTheta.at(i++);
				}
			}
		}
	}
};

#endif