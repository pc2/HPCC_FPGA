/*
Copyright (c) 2019 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef SRC_HOST_TRANSPOSE_DATA_H_
#define SRC_HOST_TRANSPOSE_DATA_H_

/* C++ standard library headers */
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {

/**
 * @brief The Transpose specific program settings
 * 
 */
class TransposeProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of the whole matrix
     * 
     */
    uint matrixSize;

    /**
     * @brief The size of a matrix block
     * 
     */
    uint blockSize;

    /**
     * @brief Identifier of the used data handler
     * 
     */
    std::string dataHandlerIdentifier;

    /**
     * @brief Construct a new Transpose Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    TransposeProgramSettings(cxxopts::ParseResult &results);

    /**
     * @brief Get a map of the settings. This map will be used to print the final configuration.
     * 
     * @return a map of program parameters. keys are the name of the parameter.
     */
    std::map<std::string, std::string> getSettingsMap() override;

};

/**
 * @brief Data class cotnaining the data the kernel is exeucted with
 * 
 */
class TransposeData {

public:
    /**
     * @brief Vector of input matrices A
     * 
     */
    HOST_DATA_TYPE* A;

    /**
     * @brief Vector of input matrices B
     * 
     */
    HOST_DATA_TYPE* B;

    /**
     * @brief Vector of the result matrices
     * 
     */
    HOST_DATA_TYPE* result;

    /**
     * @brief Number of matrix blocks that are stored in every matrix A, B and result. Blocks are
     *          always stored columnwise.
     * 
     */
    uint numBlocks;

    /**
     * @brief The width and heigth of the used blocks in number of values
     * 
     */
    uint blockSize;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief Construct a new Transpose Data object
     * 
     * @param context Context that is used to allocate memory for SVM
     * @param block_size size of the quadratic blocks that are stored within this object
     * @param y_size number of blocks that are stored within this object per replication
     */
    TransposeData(cl::Context context, uint block_size, uint size_y);

    /**
     * @brief Destroy the Transpose Data object. Free the allocated memory
     * 
     */
    ~TransposeData();

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class TransposeExecutionTimings {
public:
    /**
     * @brief A vector containing the timings for all repetitions for the data transfer
     * 
     */
    std::vector<double> transferTimings;

    /**
     * @brief A vector containing the timings for all repetitions for the calculation
     * 
     */
    std::vector<double> calculationTimings;

};

}

#endif
