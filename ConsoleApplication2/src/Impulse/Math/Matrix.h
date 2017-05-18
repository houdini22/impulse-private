#ifndef MATRIX_H
#define MATRIX_H

typedef std::vector<std::vector<double>> TypeMatrix;

#include "Vector.h"

namespace Impulse {
    namespace Math {

        class Matrix {
        public:

            static void rollMatrixToVector(TypeMatrix matrix, TypeVector & vector) {
                int xSize = matrix.size();
                int ySize = matrix.at(0).size();
                int vectorSize = xSize * ySize;

                vector.reserve(vectorSize);

                for (int i = 0; i < matrix.size(); i++) {
                    for (int j = 0; j < matrix.at(i).size(); j++) {
                        vector.push_back(matrix.at(i).at(j));
                    }
                }
            }
        };
    }
};


#endif /* MATRIX_H */

