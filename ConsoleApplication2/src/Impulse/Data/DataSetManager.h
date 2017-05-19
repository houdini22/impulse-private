#ifndef MANAGER_H
#define MANAGER_H

#include <map>

#include "DataSet.h"
#include "DataSetSample.h"

class DataSetManager {
public:

    DataSet createSet(TypeMatrix input, TypeMatrix output) {
        DataSet set = DataSet();

        for (int i = 0; i < input.size(); i++) {
            
			TypeVector newOutput;
            double realValue = output.at(i).at(0);
            for (int j = 1; j <= 10; j++) {
                newOutput.push_back(j == realValue ? 1.0 : 0.0);
            }
			

            MapSample dataSample = MapSample();
            dataSample["input"] = input.at(i);
            dataSample["output"] = newOutput;
			// dataSample["output"] = output.at(i);

            set.addSample(dataSample);
        }

        return set;
    }
};

#endif /* MANAGER_H */

