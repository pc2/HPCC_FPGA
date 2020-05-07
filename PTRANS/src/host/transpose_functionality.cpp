//
// Created by Marius Meyer on 04.12.19.
//

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

#include "transpose_functionality.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "cxxopts.hpp"
#include "setup/fpga_setup.hpp"
#include "parameters.h"

/**
Parses and returns program options using the cxxopts library.
Supports the following parameters:
    - file name of the FPGA kernel file (-f,--file)
    - number of repetitions (-n)
    - number of kernel replications (-r)
    - data size (-d)
    - use memory interleaving
@see https://github.com/jarro2783/cxxopts

@return program settings that are created from the given program arguments
*/
std::shared_ptr<ProgramSettings>
parseProgramParameters(int argc, char *argv[]) {
    // Defining and parsing program options
    cxxopts::Options options(argv[0], PROGRAM_DESCRIPTION);
    options.add_options()
            ("f,file", "Kernel file name", cxxopts::value<std::string>())
            ("n", "Number of repetitions",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_REPETITIONS)))
            ("m", "Matrix size in number of blocks in one dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
            ("b", "Block size in number of values in one dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(BLOCK_SIZE)))
            ("kernel", "Name of the kernel",
             cxxopts::value<std::string>()->default_value(KERNEL_NAME))
            ("i,nointerleaving", "Disable memory interleaving")
            ("device", "Index of the device that has to be used. If not given you "\
        "will be asked which device to use if there are multiple devices "\
        "available.", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_DEVICE)))
            ("platform", "Index of the platform that has to be used. If not given "\
        "you will be asked which platform to use if there are multiple "\
        "platforms available.",
             cxxopts::value<int>()->default_value(std::to_string(DEFAULT_PLATFORM)))
            ("h,help", "Print this help");
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("h")) {
        // Just print help when argument is given
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Check parsed options and handle special cases
    if (result.count("f") <= 0) {
        // Path to the kernel file is mandatory - exit if not given!
        std::cerr << "Kernel file must be given! Aborting" << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }

    // Create program settings from program arguments
    std::shared_ptr<ProgramSettings> sharedSettings(
            new ProgramSettings{result["n"].as<uint>(),
                                result["m"].as<cl_uint>(),
                                result["b"].as<cl_uint>(),
                                result["platform"].as<int>(),
                                result["device"].as<int>(),
                                static_cast<bool>(result.count("i") <= 0),
                                result["f"].as<std::string>(),
                                result["kernel"].as<std::string>()});
    return sharedSettings;
}

/**
 * Reference implementation that takes two matrices and calculates
 *  A_out = trans(A) + B
 *  where A, B and A_out are matrices of size n*n.
 *
 * @param A matrix that has to be transposed
 * @param B matrix that will be added to the transposed matrix
 * @param A_out matrix where the result of the calculation is stored
 * @param n dimension of the matrices
 */
void
transposeReference(HOST_DATA_TYPE *const A, HOST_DATA_TYPE *const B,
                   HOST_DATA_TYPE *A_out, cl_uint n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_out[i * n + j] = A[j * n + i] + B[i * n + j];
        }
    }
}


void
generateInputData(cl_uint matrix_size, HOST_DATA_TYPE *A, HOST_DATA_TYPE *B) {
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i * matrix_size + j] = dis(gen);
            B[j * matrix_size + i] = -A[i * matrix_size + j] + 1.0;
        }
    }
}


/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(std::shared_ptr<bm_execution::ExecutionTimings> results, cl_uint matrixSize) {
    double flops = matrixSize * matrixSize;

    double avgTransferTime = accumulate(results->transferTimings.begin(), results->transferTimings.end(), 0.0)
                             / results->transferTimings.size();
    double minTransferTime = *min_element(results->transferTimings.begin(), results->transferTimings.end());


    double avgCalculationTime = accumulate(results->calculationTimings.begin(), results->calculationTimings.end(), 0.0)
                                / results->calculationTimings.size();
    double minCalculationTime = *min_element(results->calculationTimings.begin(), results->calculationTimings.end());

    double avgCalcFLOPS = flops / avgCalculationTime;
    double avgTotalFLOPS = flops / (avgCalculationTime + avgTransferTime);
    double minCalcFLOPS = flops / minCalculationTime;
    double minTotalFLOPS = flops / (minCalculationTime + minTransferTime);

    std::cout << "             trans          calc    calc FLOPS   total FLOPS" << std::endl;
    std::cout << "avg:   " << avgTransferTime
              << "   " << avgCalculationTime
              << "   " << avgCalcFLOPS
              << "   " << avgTotalFLOPS
              << std::endl;
    std::cout << "best:  " << minTransferTime
              << "   " << minCalculationTime
              << "   " << minCalcFLOPS
              << "   " << minTotalFLOPS
              << std::endl;
}

/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings retrieved from the command line
 * @param device The device used for execution
 */
void printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                             const cl::Device &device) {// Give setup summary
    std::cout << PROGRAM_DESCRIPTION << std::endl << HLINE;
    std::cout << "Summary:" << std::endl
              << "Repetitions:         " << programSettings->numRepetitions
              << std::endl
              << "Matrix Size:         " << programSettings->matrixSize * programSettings->blockSize
              << std::endl
              << "Memory Interleaving: " << (programSettings->useMemInterleaving ? "Yes" : "No")
              << std::endl
              << "Kernel file:         " << programSettings->kernelFileName
              << std::endl;
    std::cout << "Device:              "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << HLINE
              << "Start benchmark using the given configuration." << std::endl
              << HLINE;
}

double
printCalculationError(cl_uint matrixSize, const HOST_DATA_TYPE *result) {
    double max_error = 0.0;
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        max_error = std::max(fabs(result[i] - 1.0), max_error);
    }

    std::cout << "Maximum error: " << max_error << std::endl;

    return max_error;
}