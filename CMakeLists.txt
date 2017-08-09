cmake_minimum_required(VERSION 3.7)
project(impulse)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/BaseBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/BaseBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Callback.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Callback.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Category.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Category.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MissingData.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MissingData.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/BaseDatasetModifier.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/BaseDatasetModifier.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSplitter.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSplitter.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetSample.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetSample.h
        src/Impulse/NeuralNetwork/Layer/AbstractLayer.h
        src/Impulse/NeuralNetwork/Layer/LogisticLayer.h
        src/Impulse/NeuralNetwork/Layer/InputLayer.h
        src/Impulse/NeuralNetwork/Layer/OutputLayer.h
        src/Impulse/NeuralNetwork/Layer/PurelinLayer.h
        src/Impulse/NeuralNetwork/Math/Minimizer/Fmincg.cpp
        src/Impulse/NeuralNetwork/Math/Minimizer/Fmincg.h
        src/Impulse/NeuralNetwork/Math/Matrix.h
        src/Impulse/NeuralNetwork/Math/Vector.h
        src/Impulse/NeuralNetwork/Network.h
        src/Impulse/NeuralNetwork/NetworkSerializer.h
        src/Impulse/NeuralNetwork/Neuron/AbstractNeuron.h
        src/Impulse/NeuralNetwork/Neuron/BiasNeuron.h
        src/Impulse/NeuralNetwork/Neuron/InputNeuron.h
        src/Impulse/NeuralNetwork/Neuron/LinearNeuron.h
        src/Vendor/json.hpp
        main.cpp src/Impulse/NeuralNetwork/Builder/AbstractBuilder.h src/Impulse/NeuralNetwork/Builder/RegressionBuilder.h src/Impulse/NeuralNetwork/Builder/ClassificationBuilder.h src/Impulse/NeuralNetwork/Trainer/AbstractTrainer.h src/Impulse/NeuralNetwork/Trainer/AbstractTrainer.cpp src/Impulse/NeuralNetwork/Trainer/ClassificationTrainer.h)

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -m64 -Ofast -flto -march=native -funroll-loops -mavx -msse4.2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(impulse ${SOURCE_FILES})

target_link_libraries(impulse ${Boost_LIBRARIES})
