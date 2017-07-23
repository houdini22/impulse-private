#ifndef MATRIX_H
#define MATRIX_H

typedef std::vector<std::vector<double>> TypeMatrix;

#include "Vector.h"

namespace Impulse {
    namespace Math {

        class Matrix {
        public:

            static void rollMatrixToVector(Eigen::MatrixXd & matrix, TypeVector & vector) {
				int xSize = matrix.cols();
				int ySize = matrix.rows();
                int vectorSize = xSize * ySize;

                vector.reserve(vectorSize);

				for (int i = 0; i < xSize; i++) {
					Eigen::RowVectorXd row = matrix.col(i);
					for (int j = 0; j < ySize; j++) {
						vector.push_back(row(j));
					}
				}
            }
        };
    }
};


#endif /* MATRIX_H */

