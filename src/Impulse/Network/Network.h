#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "../Layer/AbstractLayer.h"

typedef std::vector<Impulse::NeuralNetwork::Layer::AbstractLayer *> LayerContainer;

class Network {
protected:

    int size = 0;
    LayerContainer *layers;

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

    void addLayer(Impulse::NeuralNetwork::Layer::AbstractLayer *layer) {
        this->size++;
        this->layers->push_back(layer);
    }

    int getSize() {
        return this->size;
    }

    LayerContainer *getLayers() {
        return this->layers;
    }

    Eigen::VectorXd forward(Eigen::VectorXd input) {
        Eigen::VectorXd output = Eigen::VectorXd(input);
        for (LayerContainer::iterator it = this->layers->begin(); it != this->layers->end(); ++it) {
            output = (*it)->forward(output);
        }
        return output;
    }

    void backward(Eigen::VectorXd predictions, Eigen::VectorXd output) {
        Eigen::VectorXd sigma(predictions.size());
        for (int i = 0; i < predictions.size(); i++) {
            sigma(i) = predictions(i) - output(i);
        }

        for (int i = this->size - 2; i > 0; i--) {
            auto *layer = this->layers->at(i);
            auto *prevLayer = this->layers->at(i - 1);

            layer->calculateDeltas(prevLayer->getA(), sigma);
            sigma = prevLayer->backward(sigma, layer);
        }
    }

    Eigen::VectorXd getRolledTheta() {
        std::vector<double> tmp;
        for (auto layerIterator = this->layers->begin() + 1;
             layerIterator != this->layers->end() - 1; layerIterator++) {
            for (NeuronContainer::iterator neuronIterator = (*layerIterator)->getNeurons()->begin() + 1;
                 neuronIterator != (*layerIterator)->getNeurons()->end(); ++neuronIterator) {
                Eigen::VectorXd *weights = (*neuronIterator)->weights;
                for (int j = 0; j < weights->size(); j++) {
                    tmp.push_back(((*weights)(j)));
                }
            }
        }
        Eigen::Map<Eigen::VectorXd> result(tmp.data(), tmp.size());
        return result;
    }

    void setRolledTheta(Eigen::VectorXd rolledTheta) {
        int i = 0;
        for (LayerContainer::iterator it = this->getLayers()->begin() + 1; it != this->getLayers()->end() - 1; ++it) {
            for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1;
                 it2 != (*it)->getNeurons()->end(); ++it2) {
                Eigen::VectorXd *weights = (*it2)->weights;
                for (int j = 0; j < weights->size(); j++) {
                    (*weights)(j) = rolledTheta(i++);
                }
            }
        }
    }
};

#endif
