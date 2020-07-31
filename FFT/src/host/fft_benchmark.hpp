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

#ifndef SRC_HOST_FFT_BENCHMARK_H_
#define SRC_HOST_FFT_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

/**
 * @brief Contains all classes and methods needed by the FFT benchmark
 * 
 */
namespace fft {

/**
 * @brief The FFT specific program settings
 * 
 */
class FFTProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief Number of batched FFTs
     * 
     */
    uint iterations;

    /**
    * @brief Calculate inverste FFT
    */
    bool inverse;

    /**
     * @brief The number of used kernel replications
     * 
     */
    uint kernelReplications;

    /**
     * @brief Construct a new FFT Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    FFTProgramSettings(cxxopts::ParseResult &results);

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
class FFTData {

public:

    /**
     * @brief The data array used as input of the FFT calculation
     * 
     */
    std::complex<HOST_DATA_TYPE>* data;

    /**
     * @brief The data array used as output of the FFT calculation
     * 
     */
    std::complex<HOST_DATA_TYPE>* data_out;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    cl::Context context;

    /**
     * @brief Construct a new FFT Data object
     * 
     * @param context The OpenCL context used to allocate memory in SVM mode
     * @param iterations Number of FFT data that will be stored sequentially in the array
     */
    FFTData(cl::Context context, uint iterations);

    /**
     * @brief Destroy the FFT Data object. Free the allocated memory
     * 
     */
     ~FFTData();

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class FFTExecutionTimings {
public:
    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution
     * 
     */
    std::vector<double> timings;

};

/**
 * @brief Implementation of the FFT benchmark
 * 
 */
class FFTBenchmark : public hpcc_base::HpccFpgaBenchmark<FFTProgramSettings, FFTData, FFTExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the FFT benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief FFT specific implementation of the data generation
     * 
     * @return std::unique_ptr<FFTData> The input and output data of the benchmark
     */
    std::unique_ptr<FFTData>
    generateInputData() override;

    /**
     * @brief FFT specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<FFTExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<FFTExecutionTimings>
    executeKernel(FFTData &data) override;

    /**
     * @brief FFT specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutputAndPrintError(FFTData &data) override;

    /**
     * @brief FFT specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    printResults(const FFTExecutionTimings &output) override;

    /**
     * @brief Construct a new FFT Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    FFTBenchmark(int argc, char* argv[]);

        /**
     * @brief Construct a new FFT Benchmark object
     */
    FFTBenchmark();

};

/**
 * Bit reverses the order of the given FFT data in place
 *
 * @param data Array of complex numbers that will be sorted in bit reversed order
 * @param iterations Length of the data array will be calculated with iterations * FFT Size
 */
void bit_reverse(std::complex<HOST_DATA_TYPE> *data, unsigned iterations);

// The function definitions and implementations below this comment are taken from the
// FFT1D example implementation of the Intel FPGA SDK for OpenCL 19.4
// They are licensed under the following conditions:
//
// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

/**
 * @brief Do a FFT with a reference implementation on the CPU
 * 
 * @param inverse if false, the FFT will be calculated, else the iFFT
 * @param lognr_points The log2 of the FFT size that should be calculated 
 * @param data The input data for the FFT
 */
void fourier_transform_gold(bool inverse, const int lognr_points, std::complex<HOST_DATA_TYPE> *data);

/**
 * @brief Calculate a single stage of the whole FFT calculation.
 *          This function will mainly be used by fourier_transform_gold() to calculate the FFT.
 * 
 * @param lognr_points The log2 of the FFT size that should be calculated 
 * @param data The input data for the FFT stage
 */
void fourier_stage(int lognr_points, std::complex<double> *data);

} // namespace fft


#endif // SRC_HOST_STREAM_BENCHMARK_H_
