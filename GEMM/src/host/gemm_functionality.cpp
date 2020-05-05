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

/* Related header files */
#include "gemm_functionality.hpp"

/* C++ standard library headers */
#include <iostream>
#include <cmath>
#include <string>
#include <limits>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

/* External library headers */
#include "CL/cl.hpp"
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif
#include "cxxopts.hpp"

/* Project's headers */
#include "parameters.h"
#include "execution.h"


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
parseProgramParameters(int argc, char * argv[]) {
    // Defining and parsing program options
    cxxopts::Options options(argv[0], PROGRAM_DESCRIPTION);
    options.add_options()
            ("f,file", "Kernel file name", cxxopts::value<std::string>())
            ("n", "Number of repetitions",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_REPETITIONS)))
            ("m", "Matrix size",
             cxxopts::value<cl_uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
            ("kernel", "Name of the kernel",
             cxxopts::value<std::string>()->default_value(KERNEL_NAME))
#ifdef INTEL_FPGA
            ("i,interleaving", "Use memory interleaving on the FPGA")
#endif
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
                                result["platform"].as<int>(),
                                result["device"].as<int>(),
#ifdef INTEL_FPGA
                                static_cast<bool>(result.count("i") > 0),
#else
                                false,
#endif
                                result["f"].as<std::string>(),
                                result["kernel"].as<std::string>()});
    return sharedSettings;
}

/**
Print the benchmark Results

@param results The result struct provided by the calculation call
@param dataSize The size of the used data array

*/
void printResults(std::shared_ptr<bm_execution::ExecutionTimings> results,
                  size_t dataSize) {
    std::cout << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tmin = std::numeric_limits<double>::max();

    double gflops = 2.0 * static_cast<double>(dataSize*dataSize*dataSize)/1.0e9;
    for (double currentTime : results->calculationTimings) {
        tmean +=  currentTime;
        if (currentTime < tmin) {
            tmin = currentTime;
        }
    }
    tmean = tmean / results->calculationTimings.size();

    std::cout << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << gflops / tmin
              << std::endl;
}

void matgen(HOST_DATA_TYPE* a, int seed, cl_int lda, cl_int n,
            HOST_DATA_TYPE* norma) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            a[lda*i+j] = dis(gen);
            *norma = (a[lda*i+j] > *norma) ? a[lda*i+j] : *norma;
        }
        for (int i = n; i < lda; i++) {
            a[lda*j+i] = 0;
        }
    }
}

void gemm_ref(HOST_DATA_TYPE* a,HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
                                int n, HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta) {
#ifdef _USE_BLAS_
    char ta = 'N';
    char tb = 'N';
    
    sgemm_(&ta, &tb, &n, &n, &n, &alpha, b, &n, a, &n, &beta, c, &n);
#else
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            c[i * n + j] = beta * c[i*n + j];
        }
    }

    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            for (int k=0; k < n; k++) {
                c[i*n + j] += alpha * a[i*n + k] * b[k*n + j];
            }
        }
    }
#endif
}

double
checkGEMMresults(HOST_DATA_TYPE* c_res, cl_int lda, cl_int n) {
    HOST_DATA_TYPE* a = new HOST_DATA_TYPE[lda*n];
    HOST_DATA_TYPE* b = new HOST_DATA_TYPE[lda*n];
    HOST_DATA_TYPE* c = new HOST_DATA_TYPE[lda*n];
    HOST_DATA_TYPE totalnorm = 0.0;

    /*     compute a residual to verify results.  */


    matgen(a, 1, lda, n, &totalnorm);
    matgen(b, 2, lda, n, &totalnorm);
    matgen(c, 3, lda, n, &totalnorm);

    gemm_ref(a, b, c, n, 0.5, 2.0);

    HOST_DATA_TYPE resid = 0.0;
    HOST_DATA_TYPE normx = 0.0;

    for (int i = 0; i < n * n; i++) {
        resid = (resid > fabs(c_res[i] - c[i])) ? resid : fabs(c_res[i] - c[i]);
        normx = (normx > fabs(c_res[i])) ? normx : fabs(c_res[i]);
    }

    HOST_DATA_TYPE eps = epslon(static_cast<HOST_DATA_TYPE>(1.0));
    HOST_DATA_TYPE residn = resid / (lda*n*totalnorm*normx*eps);

    std::cout << "  norm. resid        resid       "\
                 "machep" << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << residn << std::setw(ENTRY_SPACE)
              << resid << std::setw(ENTRY_SPACE) << eps
              << std::endl;

    delete a;
    delete b;
    delete c;
    return residn;
}

HOST_DATA_TYPE epslon(HOST_DATA_TYPE x) {
    HOST_DATA_TYPE a, b, c, eps;

    a = 4.0e0/3.0e0;
    eps = 0.0;
    while (eps == 0.0) {
        b = a - 1.0;
        c = b + b + b;
        eps = fabs(static_cast<double>(c-1.0));
    }
    return (eps*fabs(static_cast<double>(x)));
}


