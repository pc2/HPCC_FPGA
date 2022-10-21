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
#include "data_handlers/data_handler_types.h"


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
     * @brief Value of P defining the PQ grid used to order FPGAs
     * 
     */
    uint p;

    /**
     * @brief Identifier of the used data handler
     * 
     */
    transpose::data_handler::DataHandlerType dataHandlerIdentifier;

    /**
     * @brief If true, the three buffers for A,B and A_out will be placed on three different memory banks, if possible
     *          instead of a single one
     */
    bool distributeBuffers;

    /**
    * @brief If true, create a copy of matrix A for each kernel replication
    *
    */
    bool copyA;

    /**
     * @brief Indicate, if a design is used where the user kernels are directly connected to the ACCL CCLO
    */
    bool useAcclStreams;

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
template<class TContext>
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
     * @brief Data buffer used during data exchange of matrices
     * 
     */
    HOST_DATA_TYPE* exchange;

    /**
     * @brief Number of matrix blocks that are stored in every matrix A, B and result. Blocks are
     *          always stored columnwise.
     * 
     */
    const size_t numBlocks;

    /**
     * @brief The width and heigth of the used blocks in number of values
     * 
     */
    const size_t blockSize;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    TContext context;

    /**
     * @brief Construct a new Transpose Data object
     * 
     * @param context Context that is used to allocate memory for SVM
     * @param block_size size of the quadratic blocks that are stored within this object
     * @param y_size number of blocks that are stored within this object per replication
     */
    TransposeData(TContext context, uint block_size, uint y_size): context(context),
                                                                   numBlocks(y_size), blockSize(block_size) {
        if (numBlocks * blockSize > 0) {
#ifdef USE_SVM
            A = reinterpret_cast<HOST_DATA_TYPE*>(
                                     clSVMAlloc(context(), 0 ,
                                 block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 4096));
            B = reinterpret_cast<HOST_DATA_TYPE*>(
                                clSVMAlloc(context(), 0 ,
                                block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 4096));
            result = reinterpret_cast<HOST_DATA_TYPE*>(
                                clSVMAlloc(context(), 0 ,
                                block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 4096));
            exchange = reinterpret_cast<HOST_DATA_TYPE*>(
                                clSVMAlloc(context(), 0 ,
                                block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 4096));
#else
            posix_memalign(reinterpret_cast<void **>(&A), 4096,
                        sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
            posix_memalign(reinterpret_cast<void **>(&B), 4096,
                        sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
            posix_memalign(reinterpret_cast<void **>(&result), 4096,
                        sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
            posix_memalign(reinterpret_cast<void **>(&exchange), 4096,
                        sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
#endif
        }
    }

    /**
     * @brief Destroy the Transpose Data object. Free the allocated memory
     * 
     */
    ~TransposeData() {
        if (numBlocks * blockSize > 0) {
#ifdef USE_SVM
            clSVMFree(context(), reinterpret_cast<void*>(A));});
            clSVMFree(context(), reinterpret_cast<void*>(B));});
            clSVMFree(context(), reinterpret_cast<void*>(result));});
            clSVMFree(context(), reinterpret_cast<void*>(exchange));});
#else
            free(A);
            free(B);
            free(result);
            free(exchange);
#endif
        }
    }

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
