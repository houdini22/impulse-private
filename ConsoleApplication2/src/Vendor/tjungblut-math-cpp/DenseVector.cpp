#include "stdafx.h"

#pragma once

#include <string.h>
#include <math.h>
#include <iostream>
#include <ctime>
#include "DenseVector.h"

#define NUM_THREAD 6

namespace tjmath {

	template <class T>
	DenseVector<T>::DenseVector(void)
	{
		this->dimension = 0;
		this->data = NULL;
	};

	template <class T>
	DenseVector<T>::DenseVector(DenseVector<T>& vx)
	{
		this->dimension = vx.getDimension();
		this->data = new T[this->dimension];
		memcpy(this->data, vx.data, this->dimension * sizeof(T));
	};

	template <class T>
	DenseVector<T>::DenseVector(unsigned int length, T defaultValue)
	{
		this->dimension = length;
		this->data = new T[length];
		for (unsigned int i = 0; i < this->dimension; i++) 
		{
			this->data[i] = defaultValue;
		}
	};

	template <class T>
	DenseVector<T>::DenseVector(unsigned int length, T* lst)
	{
		this->dimension = length;
		this->data = new T[length];
		memcpy(this->data, lst, length * sizeof(T));
	};

	template <class T>
	DenseVector<T>::DenseVector(unsigned int length, std::vector<T> lst)
	{
		this->dimension = length;
		this->data = new T[length];
		memcpy(this->data, lst.data(), length * sizeof(T));
	};

	template <class T>
	DenseVector<T>::~DenseVector(void)
	{
		if(this->data != NULL){
			delete [] this->data;
		}
	};

	template <class T>
	void DenseVector<T>::set(unsigned int index, T value)
	{
		this->data[index] = value;
	};

	template <class T>
	T DenseVector<T>::get(unsigned int index)
	{
		return this->data[index];
	};

	/*
	template <class T>
	unsigned int DenseVector<T>::getLength()
	{
		return this->dimension;
	};
	*/

	template <class T>
	unsigned int DenseVector<T>::getDimension()
	{
		return this->dimension;
	};


	/*
	template <class T>
	VectorIterator<T> DenseVector<T>::begin()
	{
		return VectorIterator<T>(this->data);
	};

	template <class T>
	VectorIterator<T> DenseVector<T>::end()
	{
		return VectorIterator<T>(&this->data[this->dimension]);
	};
	*/

	template <class T>
	DenseVector<T> DenseVector<T>::operator+(DenseVector<T> v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		clock_t begin = clock();
		int index = 0;
#pragma omp parallel for shared(newVector, v) private(index) schedule(dynamic)
		for(index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] + v.data[index];
		}
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Time: " << elapsed_secs << std::endl;
		return newVector;
	};

	template <class T>
	DenseVector<T> DenseVector<T>::operator+(T v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] + v;
		}

		return newVector;
	};

	template <class T>
	DenseVector<T> DenseVector<T>::operator-(DenseVector<T> v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] - v.get(index);
		}
		return newVector;
	};


	template <class T>
	DenseVector<T> DenseVector<T>::operator-(T v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] - v;
		}

		return newVector;
	};

	/*
	template <class T>
	DenseVector<T> DenseVector<T>::subtractFrom(T scalar)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = scalar- newVector.data[index];
		}

		return newVector;
	};
	*/

	template <class T>
	DenseVector<T> DenseVector<T>::operator*(DenseVector<T> v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] * v.get(index);
		}
		return newVector;
	};


	template <class T>
	DenseVector<T> DenseVector<T>::operator*(T v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] * v;
		}

		return newVector;
	};

	template <class T>
	DenseVector<T> DenseVector<T>::operator/(DenseVector<T> v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] / v.get(index);
		}
		return newVector;
	};


	template <class T>
	DenseVector<T> DenseVector<T>::operator/(T v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = newVector.data[index] / v;
		}

		return newVector;
	};

	/*
	template <class T>
	DenseVector<T> DenseVector<T>::divideFrom(DenseVector<T> v)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = v.data[index] / newVector.data[index];
		}

		return newVector;
	};

	template <class T>
	DenseVector<T> DenseVector<T>::divideFrom(T scalar)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = scalar / newVector.data[index];
		}

		return newVector;
	};
	*/

	template <class T>
	void DenseVector<T>::operator<<(DenseVector<T> newValues)
	{
		memcpy(this->data, newValues.data, this->dimension * sizeof(T));
	};

	/*
	template <class T>
	DenseVector<T> DenseVector<T>::pow(T scalar)
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = std::pow(newVector.data[index], scalar);
		}

		return newVector;
	}

	template <class T>
	DenseVector<T> DenseVector<T>::sqrt()
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = std::sqrt(newVector.data[index]);
		}

		return newVector;
	}

	template <class T>
	DenseVector<T> DenseVector<T>::log()
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = std::log(newVector.data[index]);
		}

		return newVector;
	}

	template <class T>
	DenseVector<T> DenseVector<T>::exp()
	{
		DenseVector<T> newVector(this->dimension, this->data);

		for(unsigned int index = 0; index < this->dimension; index++)
		{
			newVector.data[index] = std::exp(newVector.data[index]);
		}

		return newVector;
	}

	template <class T>
	T DenseVector<T>::sum()
	{
		T val = this->data[0];
		for(unsigned int index = 1; index < this->dimension; index++)
		{
			val += this->data[index];
		}
		return val;
	}
	*/

	template <class T>
	T DenseVector<T>::dot(DenseVector<T> v)
	{
		T val = this->data[0] * v.get(0);
		for(unsigned int index = 1; index < this->dimension; index++)
		{
			val += (this->data[index] * v.get(index));
		}
		return val;
	}

	/*
	template <class T>
	unsigned int DenseVector<T>::minIndex()
	{
		unsigned int minIndex = 0;
		for(unsigned int index = 1; index < this->dimension; index++)
		{
			if(this->data[minIndex] > this->data[index])
			{
				minIndex = index;
			}
		}
		return minIndex;
	}

	template <class T>
	T DenseVector<T>::min()
	{
		return this->data[minIndex()];
	}

	template <class T>
	unsigned int DenseVector<T>::maxIndex()
	{
		unsigned int maxIndex = 0;
		for(unsigned int index = 1; index < this->dimension; index++)
		{
			if(this->data[maxIndex] < this->data[index])
			{
				maxIndex = index;
			}
		}
		return maxIndex;
	}

	template <class T>
	T DenseVector<T>::max()
	{
		return this->data[maxIndex()];
	}
	*/

	template class DenseVector<double>;
}