#ifndef MATRIX_H
#define MATRIX_H

typedef std::vector<std::vector<double>> TypeMatrix;

#include "Vector.h"

namespace Impulse {
    namespace Math {

        class Matrix {
        public:

            static void rollMatrixToVector(Eigen::MatrixXd &matrix, TypeVector &vector) {
                unsigned int xSize = matrix.cols();
                unsigned int ySize = matrix.rows();
                unsigned int vectorSize = xSize * ySize;

                vector.reserve(vectorSize);

                for (unsigned int i = 0; i < xSize; i++) {
                    Eigen::RowVectorXd row = matrix.col(i);
                    for (unsigned int j = 0; j < ySize; j++) {
                        vector.push_back(row(j));
                    }
                }
            }
        };
    }
};


#endif /* MATRIX_H */

