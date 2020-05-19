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

#ifndef SRC_HOST_LINPACK_BENCHMARK_H_
#define SRC_HOST_LINPACK_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

/**
 * @brief Contains all classes and methods needed by the LINPACK benchmark
 * 
 */
namespace linpack {

/**
 * @brief The Linpack specific program settings
 * 
 */
class LinpackProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of the whole matrix
     * 
     */
    uint matrixSize;

    /**
     * @brief Construct a new Linpack Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    LinpackProgramSettings(cxxopts::ParseResult &results);

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
class LinpackData {

public:
    HOST_DATA_TYPE *A, *B;
    cl_int* ipvt;

    /**
     * @brief Construct a new Linpack Data object
     * 
     * @param size Size of the allocated square matrix and vectors
     */
    LinpackData(uint size) {
        posix_memalign(reinterpret_cast<void**>(&A), 4096, size * size * sizeof(HOST_DATA_TYPE));
        posix_memalign(reinterpret_cast<void**>(&b), 4096, size * sizeof(HOST_DATA_TYPE));
        posix_memalign(reinterpret_cast<void**>(&ipvt), 4096, size * sizeof(cl_int));
    }

    /**
     * @brief Destroy the Linpack Data object. Free the allocated memory
     * 
     */
    ~LinpackData() {
        free(A);
        free(B);
        free(ipvt);
    }

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class LinpackExecutionTimings {
public:
    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution
     * 
     */
    std::vector<double> timings;

};

/**
 * @brief Implementation of the Linpack benchmark
 * 
 */
class LinpackBenchmark : public hpcc_base::HpccFpgaBenchmark<LinpackProgramSettings, LinpackData, LinpackExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the Linpack benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief LINPACK specific implementation of the data generation
     * 
     * @return std::unique_ptr<LinpackData> The input and output data of the benchmark
     */
    std::unique_ptr<LinpackData>
    generateInputData() override;

    /**
     * @brief Linpack specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<LinpackExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<LinpackExecutionTimings>
    executeKernel(LinpackData &data) override;

    /**
     * @brief Linpack specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutputAndPrintError(LinpackData &data) override;

    /**
     * @brief Linpack specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    printResults(const LinpackExecutionTimings &output) override;

    /**
     * @brief Construct a new Linpack Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    LinpackBenchmark(int argc, char* argv[]);

        /**
     * @brief Construct a new Linpack Benchmark object
     */
    LinpackBenchmark();

};

/**
 *
 *
 * @param n1
 * @param y
 * @param n2
 * @param ldm
 * @param x
 * @param m
 */
void dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m);

/**
Gaussian elemination reference implementation without pivoting.
Can be used in exchange with kernel functions for functionality testing

@param a the matrix with size of n*n
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gefa_ref(HOST_DATA_TYPE* a, unsigned n, unsigned lda, cl_int* ipvt);

/**
Solve linear equations using its LU decomposition.
Therefore solves A*x = b by solving L*y = b and then U*x = y with A = LU
where A is a matrix of size n*n

@param a the matrix a in LU representation calculated by gefa call
@param b vector b of the given equation
@param ipvt vector containing pivoting information
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gesl_ref(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, cl_int* ipvt, unsigned n, unsigned lda);

} // namespace stream


#endif // SRC_HOST_STREAM_BENCHMARK_H_
