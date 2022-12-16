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

#ifndef SRC_HOST_RANDOM_ACCESS_BENCHMARK_H_
#define SRC_HOST_RANDOM_ACCESS_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

/**
 * @brief Contains all classes and methods needed by the RandomAccess benchmark
 * 
 */
namespace random_access {

/**
 * @brief The random access specific program settings
 * 
 */
class RandomAccessProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of the data array
     * 
     */
    size_t dataSize;

    /**
     * @brief Number of random number generators that are used per kernel replication
     * 
     */
    uint numRngs;

    /**
     * @brief Construct a new random access Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    RandomAccessProgramSettings(cxxopts::ParseResult &results);

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
class RandomAccessData {

public:

    /**
     * @brief The input data array that will be updated using random accesses
     * 
     */
    HOST_DATA_TYPE *data;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief Construct a new Random Access Data object
     * 
     * @param context The OpenCL context that will be used to allocate SVM memory
     * @param size The size  of the allocated memory in number of values
     */
    RandomAccessData(cl::Context& context, size_t size);

    /**
     * @brief Destroy the Random Access Data object and free the memory allocated in the constructor
     * 
     */
    ~RandomAccessData();

};

/**
 * @brief Implementation of the random access benchmark
 * 
 */
class RandomAccessBenchmark : public hpcc_base::HpccFpgaBenchmark<RandomAccessProgramSettings, RandomAccessData> {

protected:

    /**
     * @brief Additional input parameters of the random access benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief Random access specific implementation of the data generation
     * 
     * @return std::unique_ptr<RandomAccessData> pointer to the object storing the benchmark input and output data
     */
    std::unique_ptr<RandomAccessData>
    generateInputData() override;

    /**
     * @brief RandomAccess specific implementation of the kernel execution
     * 
     * @param data The benchmark input and output data
     */
    void
    executeKernel(RandomAccessData &data) override;

    /**
     * @brief RandomAccess specific implementation of the execution validation
     * 
     * @param data The benchmark input and output data
     * @return true If the validation is successful
     * @return false otherwise
     */
    bool
    validateOutput(RandomAccessData &data) override;

    /**
     * @brief RandomAccess specific implementation of the error printing
     *
     */
    void
    printError() override;

    /**
     * @brief RandomAccess specific implementation of printing the execution results
     * 
     * @param output The measurement values that are generated yb the kernel execution
     */
    void
    collectResults() override;

    void
    printResults() override;

    /**
     * @brief Check the given bencmark configuration and its validity
     * 
     * @return true if the validation is successful, false otherwise
     */
    bool
    checkInputParameters() override;

    /**
     * @brief Construct a new RandomAccess Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    RandomAccessBenchmark(int argc, char* argv[]);

     /**
     * @brief Construct a new RandomAccess Benchmark object
     */
    RandomAccessBenchmark();

};

} // namespace stream


#endif // SRC_HOST_STREAM_BENCHMARK_H_
