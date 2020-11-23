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

#ifndef SRC_HOST_STREAM_BENCHMARK_H_
#define SRC_HOST_STREAM_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

#include "half.hpp"

/**
 * @brief Contains all classes and methods needed by the STREAM benchmark
 * 
 */
namespace stream {

/**
 * @brief The STREAM specific program settings
 * 
 */
class StreamProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of each stream array in number of values
     * 
     */
    uint streamArraySize;

    /**
     * @brief The number of used kernel replications
     * 
     */
    uint kernelReplications;

    /**
     * @brief Indicator if the single kernel or the legacy kernel are used for execution
     * 
     */
    bool useSingleKernel;

    /**
     * @brief Construct a new Stream Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    StreamProgramSettings(cxxopts::ParseResult &results);

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
class StreamData {

public:
    /**
     * @brief The input array A of the benchmark
     * 
     */
    HOST_DATA_TYPE *A;

    /**
     * @brief The input array B of the benchmark
     * 
     */
    HOST_DATA_TYPE *B;

    /**
     * @brief The input array C of the benchmark
     * 
     */
    HOST_DATA_TYPE *C;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief Construct a new Stream Data object
     * 
     * @param _context the context that will be used to allocate SVM memory
     * @param size the size of the data arrays in number of values
     */
    StreamData(const cl::Context& _context, size_t size);

    /**
     * @brief Destroy the Stream Data object
     * 
     */
    ~StreamData();

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class StreamExecutionTimings {
public:
    /**
     * @brief A map containing the timings for all stream operation types
     * 
     */
    std::map<std::string,std::vector<double>> timings;

    /**
     * @brief The used array size
     * 
     */
    uint arraySize;
};

/**
 * @brief Implementation of the Sream benchmark
 * 
 */
class StreamBenchmark : public hpcc_base::HpccFpgaBenchmark<StreamProgramSettings, StreamData, StreamExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the strema benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief Stream specific implementation of the data generation
     *  
     * @return std::unique_ptr<StreamData> 
     */
    std::unique_ptr<StreamData>
    generateInputData() override;

    /**
     * @brief Stream specific implementation of the kernel execution
     * 
     * @param data 
     * @return std::unique_ptr<StreamExecutionTimings> 
     */
    std::unique_ptr<StreamExecutionTimings>
    executeKernel( StreamData &data) override;

    /**
     * @brief Stream specific implementation of the execution validation
     *  
     * @param data 
     * @return true 
     * @return false 
     */
    bool
    validateOutputAndPrintError(StreamData &data) override;

    /**
     * @brief Stream specific implementation of printing the execution results
     * 
     * @param output 
     */
    void
    collectAndPrintResults(const StreamExecutionTimings &output) override;

    /**
     * @brief Construct a new Stream Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    StreamBenchmark(int argc, char* argv[]);

     /**
     * @brief Construct a new Stream Benchmark object
     */
    StreamBenchmark();

};

} // namespace stream


#endif // SRC_HOST_STREAM_BENCHMARK_H_
