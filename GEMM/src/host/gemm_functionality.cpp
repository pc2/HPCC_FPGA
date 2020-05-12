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
#include "setup/common_benchmark_io.hpp"


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


