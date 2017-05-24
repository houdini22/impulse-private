#ifndef VECTOR_H
#define VECTOR_H

#include <Eigen/Core>
#include <Eigen/Dense>

typedef std::vector<double> TypeVector;

namespace Impulse {
	namespace Math {

		class Vector {
		public:

			static TypeVector eigenToVector(Eigen::VectorXd & vector) {
				TypeVector result;
				for (int i = 0; i < vector.size(); i++) {
					result.push_back(vector(i));
				}
				return result;
			}

			static Eigen::VectorXd vectorToEigen(TypeVector v) {
				Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(v.data(), v.size());
				return result;
			}
		};
	}
};

#endif /* VECTOR_H */

