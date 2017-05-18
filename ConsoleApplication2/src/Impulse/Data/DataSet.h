#ifndef SET_H
#define SET_H

#include <vector>

#include "DataSetSample.h"

class DataSet {
protected:
    int size = 0;
    std::vector<MapSample> samples;
public:

    void addSample(MapSample sample) {
        this->size++;
        this->samples.push_back(sample);
    }

    int getSize() {
        return this->size;
    }

    std::vector<MapSample> getSamples() {
        return this->samples;
    }
};

#endif /* SET_H */

