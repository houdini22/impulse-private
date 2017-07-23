#ifndef MANAGER_H
#define MANAGER_H

#include <map>

#include "DataSet.h"
#include "DataSetSample.h"

class DataSetManager {
public:

    DataSet createSet(Eigen::MatrixXd input, Eigen::MatrixXd output) {
        DataSet set = DataSet();

        for (int i = 0; i < input.rows(); i++) {
            Eigen::VectorXd newInput(input.row(i));
            Eigen::VectorXd newOutput(output.row(i));

            MapSample dataSample = MapSample();
            dataSample["input"] = newInput;
            dataSample["output"] = newOutput;

            set.addSample(dataSample);
        }

        return set;
    }
};

#endif /* MANAGER_H */

