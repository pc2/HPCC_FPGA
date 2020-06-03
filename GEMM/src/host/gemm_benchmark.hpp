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

#ifndef SRC_HOST_GEMM_BENCHMARK_H_
#define SRC_HOST_GEMM_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

#ifdef _USE_BLAS_

extern "C" void sgemm_(char*, char*, int*, int*,int*, float*, float*, int*, float*, int*, float*, float*, int*);

#endif


/**
 * @brief Contains all classes and methods needed by the GEMM benchmark
 * 
 */
namespace gemm {

/**
 * @brief The GEMM specific program settings
 * 
 */
class GEMMProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of the whole matrix
     * 
     */
    uint matrixSize;

    /**
     * @brief Construct a new GEMM Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    GEMMProgramSettings(cxxopts::ParseResult &results);

    /**
     * @brief Get a map of the settings. This map will be used to print the final configuration.
     * 
     * @return a map of program parameters. keys are the name of the parameter.
     */
    std::map<std::string, std::string> getSettingsMap() override;

};

/**
 * @brief Data class containing all data needed by the kernel to calculate
 *          \f$C\_out = \alpha * A * B + \beta * C\f$
 * 
 */
class GEMMData {

public:
    /**
     * @brief Pointer to the matrix A of the calculation
     * 
     */
    HOST_DATA_TYPE *A;

    /**
     * @brief Pointer to the matrix B of the calculation
     * 
     */
    HOST_DATA_TYPE *B;

    /**
     * @brief Pointer to the matrix C of the calculation
     * 
     */
    HOST_DATA_TYPE *C;

    /**
     * @brief Pointer to the output matrix of the calculation
     * 
     */
    HOST_DATA_TYPE *C_out;

    /**
     * @brief Stores the maximum value of all input matrices for the error calculation
     * 
     */
    HOST_DATA_TYPE normtotal;

    /**
     * @brief The scalar value that will be used for \f$\alpha\f$ in the calculation
     * 
     */
    HOST_DATA_TYPE alpha;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief The scalar value that will be used for \f$\beta\f$ in the calculation
     * 
     */
    HOST_DATA_TYPE beta;

    /**
     * @brief Construct a new GEMM Data object
     * 
     * @param context The OpenCL context used to allocate memory in SVM mode
     * @param size Size of the allocated square matrices
     */
    GEMMData(cl::Context context, uint size);

    /**
     * @brief Destroy the GEMM Data object. Free the allocated memory
     * 
     */
    ~GEMMData();

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class GEMMExecutionTimings {
public:
    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution
     * 
     */
    std::vector<double> timings;

};

/**
 * @brief Implementation of the GEMM benchmark
 * 
 */
class GEMMBenchmark : public hpcc_base::HpccFpgaBenchmark<GEMMProgramSettings, GEMMData, GEMMExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the GEMM benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief LINPACK specific implementation of the data generation
     * 
     * @return std::unique_ptr<GEMMData> The input and output data of the benchmark
     */
    std::unique_ptr<GEMMData>
    generateInputData() override;

    /**
     * @brief GEMM specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<GEMMExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<GEMMExecutionTimings>
    executeKernel(GEMMData &data) override;

    /**
     * @brief GEMM specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutputAndPrintError(GEMMData &data) override;

    /**
     * @brief GEMM specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    printResults(const GEMMExecutionTimings &output) override;

    /**
     * @brief Construct a new GEMM Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    GEMMBenchmark(int argc, char* argv[]);

        /**
     * @brief Construct a new GEMM Benchmark object
     */
    GEMMBenchmark();

};

/**
Multiply matrix with a vector and add it to another vector.

C = alpha * A * B + beta * C

@param a matrix A
@param b matrix B
@param c matrix C that will also be the result matrix
@param n size of all quadratic matrices
@param alpha scalar value used to scale A * B
@param beta scalar value used to scale C
*/
void gemm_ref( HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
                                int n, HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta);

} // namespace gemm


#endif // SRC_HOST_STREAM_BENCHMARK_H_
