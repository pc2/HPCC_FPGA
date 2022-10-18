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
#include "transpose_data.hpp"

#include "data_handlers/data_handler_types.h"
#include "data_handlers/handler.hpp"

#include "parameters.h"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {

/**
 * @brief Implementation of the transpose benchmark
 * 
 */
class TransposeBenchmark : public hpcc_base::HpccFpgaBenchmark<TransposeProgramSettings, TransposeData> {

protected:

    /**
     * @brief Additional input parameters of the transpose benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

    std::unique_ptr<transpose::data_handler::TransposeDataHandler> dataHandler;

public:

    /**
     * @brief Random access specific implementation of the data generation
     * 
     * @return std::unique_ptr<TransposeData> The input and output data of the benchmark
     */
    std::unique_ptr<TransposeData>
    generateInputData() override;

    /**
     * @brief Set the data handler object by calling the function with the matching template argument
     * 
     */
    void
    setTransposeDataHandler(transpose::data_handler::DataHandlerType dataHandlerIdentifier);

    /**
     * @brief Transpose specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     */
    void
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
    collectResults() override;
    
    void
    printResults() override;

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
