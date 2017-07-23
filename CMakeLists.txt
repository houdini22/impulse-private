cmake_minimum_required(VERSION 3.7)
project(impulse)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/Impulse/Data/DataSet.h
        src/Impulse/Data/DataSetManager.h
        src/Impulse/Data/DataSetSample.h
        src/Impulse/Layer/AbstractLayer.h
        src/Impulse/Layer/HiddenLayer.h
        src/Impulse/Layer/InputLayer.h
        src/Impulse/Layer/OutputLayer.h
        src/Impulse/Layer/PurelinLayer.h
        src/Impulse/Math/Minimizer/Fmincg.cpp
        src/Impulse/Math/Minimizer/Fmincg.h
        src/Impulse/Math/Matrix.h
        src/Impulse/Math/Vector.h
        src/Impulse/Network/Network.h
        src/Impulse/Network/NetworkBuilder.h
        src/Impulse/Network/NetworkSerializer.h
        src/Impulse/Network/NetworkTrainer.cpp
        src/Impulse/Network/NetworkTrainer.h
        src/Impulse/Neuron/AbstractNeuron.h
        src/Impulse/Neuron/BiasNeuron.h
        src/Impulse/Neuron/InputNeuron.h
        src/Impulse/Neuron/Neuron.h
        src/Vendor/json.hpp
        main.cpp)

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(impulse ${SOURCE_FILES})

target_link_libraries(impulse ${Boost_LIBRARIES})
