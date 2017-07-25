#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "../Layer/AbstractLayer.h"

typedef std::vector<Impulse::NeuralNetwork::Layer::AbstractLayer *> LayerContainer;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            class Network {
            protected:

                unsigned int size = 0;
                LayerContainer layers;

            public:

                Network() {
                }

                ~Network() {
                    for (auto i = this->layers.begin(); i != this->layers.end(); i++) {
                        delete *i;
                    }
                    this->layers.clear();
                }

                void addLayer(Impulse::NeuralNetwork::Layer::AbstractLayer *layer) {
                    this->size++;
                    this->layers.push_back(layer);
                }

                unsigned int getSize() {
                    return this->size;
                }

                LayerContainer *getLayers() {
                    return &this->layers;
                }

                Eigen::VectorXd forward(Eigen::VectorXd input) {
                    Eigen::VectorXd output = Eigen::VectorXd(input);
                    for (LayerContainer::iterator it = this->layers.begin(); it != this->layers.end(); ++it) {
                        output = (*it)->forward(output);
                    }
                    return output;
                }

                void backward(Eigen::VectorXd predictions, Eigen::VectorXd output) {
                    Eigen::VectorXd sigma(predictions.size());
                    for (unsigned int i = 0; i < predictions.size(); i++) {
                        sigma(i) = predictions(i) - output(i);
                    }

                    for (unsigned int i = this->size - 2; i > 0; i--) {
                        auto *layer = this->layers.at(i);
                        auto *prevLayer = this->layers.at(i - 1);

                        layer->calculateDeltas(prevLayer->getA(), sigma);
                        sigma = prevLayer->backward(sigma, layer);
                    }
                }

                Eigen::VectorXd getRolledTheta() {
                    Impulse::Math::TypeVector tmp;
                    for (LayerContainer::iterator it = this->getLayers()->begin() + 1;
                         it != this->getLayers()->end() - 1; ++it) {
                        for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1;
                             it2 != (*it)->getNeurons()->end(); ++it2) {
                            for (unsigned int j = 0; j < (*it2)->weights.size(); j++) {
                                tmp.push_back(((*it2)->weights(j)));
                            }
                        }
                    }
                    Eigen::Map<Eigen::VectorXd> result(tmp.data(), tmp.size());
                    return result;
                }

                void setRolledTheta(Eigen::VectorXd rolledTheta) {
                    unsigned int i = 0;
                    for (LayerContainer::iterator it = this->getLayers()->begin() + 1;
                         it != this->getLayers()->end() - 1; ++it) {
                        for (NeuronContainer::iterator it2 = (*it)->getNeurons()->begin() + 1;
                             it2 != (*it)->getNeurons()->end(); ++it2) {
                            for (unsigned int j = 0; j < (*it2)->weights.size(); j++) {
                                (*it2)->weights(j) = rolledTheta(i++);
                            }
                        }
                    }
                }
            };

        }

    }

}

#endif
