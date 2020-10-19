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

#ifndef SRC_HOST_TRANSPOSE_BENCHMARK_H_
#define SRC_HOST_TRANSPOSE_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

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
     * @brief Input matrix A
     * 
     */
    HOST_DATA_TYPE *A;

    /**
     * @brief Input matrix B
     * 
     */
    HOST_DATA_TYPE *B;

    /**
     * @brief The result matrix
     * 
     */
    HOST_DATA_TYPE *result;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief Construct a new Transpose Data object
     * 
     * @param context Context that is used to allocate memory for SVM
     * @param size size_x of the allocated square matrices
     * @param size size_y of the allocated square matrices
     */
    TransposeData(cl::Context context, uint size_x, uint size_y);

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

/**
 * @brief Implementation of the transpose benchmark
 * 
 */
class TransposeBenchmark : public hpcc_base::HpccFpgaBenchmark<TransposeProgramSettings, TransposeData, TransposeExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the transpose benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief Random access specific implementation of the data generation
     * 
     * @return std::unique_ptr<TransposeData> The input and output data of the benchmark
     */
    std::unique_ptr<TransposeData>
    generateInputData() override;

    /**
     * @brief Transpose specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<TransposeExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<TransposeExecutionTimings>
    executeKernel(TransposeData &data) override;

    /**
     * @brief Transpose specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutputAndPrintError(TransposeData &data) override;

    /**
     * @brief Transpose specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    collectAndPrintResults(const TransposeExecutionTimings &output) override;

    /**
     * @brief Construct a new Transpose Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    TransposeBenchmark(int argc, char* argv[]);

        /**
     * @brief Construct a new Transpose Benchmark object
     */
    TransposeBenchmark();

};

} // namespace stream


#endif // SRC_HOST_STREAM_BENCHMARK_H_
