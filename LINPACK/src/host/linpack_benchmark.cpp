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

#include "linpack_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

linpack::LinpackProgramSettings::LinpackProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["s"].as<uint>()) {

}

std::map<std::string, std::string>
linpack::LinpackProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        return map;
}

linpack::LinpackData::LinpackData(cl::Context context, uint size) : norma(0.0), context(context) {
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    b = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size  * sizeof(HOST_DATA_TYPE), 1024));
    ipvt = reinterpret_cast<cl_int*>(
                        clSVMAlloc(context(), 0 ,
                        size * sizeof(cl_int), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&A), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&b), 4096, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&ipvt), 4096, size * sizeof(cl_int));
#endif
    }

linpack::LinpackData::~LinpackData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void*>(A));
    clSVMFree(context(), reinterpret_cast<void*>(b));
    clSVMFree(context(), reinterpret_cast<void*>(ipvt));
#else
    free(A);
    free(b);
    free(ipvt);
#endif
}

linpack::LinpackBenchmark::LinpackBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
}

void
linpack::LinpackBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("s", "Matrix size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)));
}

std::unique_ptr<linpack::LinpackExecutionTimings>
linpack::LinpackBenchmark::executeKernel(LinpackData &data) {
    return bm_execution::calculate(*executionSettings, data.A, data.b, data.ipvt);
}

void
linpack::LinpackBenchmark::collectAndPrintResults(const linpack::LinpackExecutionTimings &output) {
     std::cout << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tmin = std::numeric_limits<double>::max();

    // GFLOPs for calculation of both GEFA and GESL.
    // Currently only GEFA is calculated on the FPGA so GFLOPS have to be
    // reduced.
    // double gflops = ((2.0e0*(dataSize*dataSize*dataSize))/3.0
    //                 + 2.0*(dataSize*dataSize)) / 1.0e9;
    // TODO: Change this when GESL is also calculated on FPGA
    double gflops = (2.0e0*(static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)))/3.0/1.0e9;
    for (double currentTime : output.timings) {
        tmean +=  currentTime;
        if (currentTime < tmin) {
            tmin = currentTime;
        }
    }
    tmean = tmean / output.timings.size();

    std::cout << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << (gflops / tmin)
              << std::endl;

}

std::unique_ptr<linpack::LinpackData>
linpack::LinpackBenchmark::generateInputData() {
    auto d = std::unique_ptr<linpack::LinpackData>(new linpack::LinpackData(*executionSettings->context ,executionSettings->programSettings->matrixSize));
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    d->norma = 0.0;
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            d->A[executionSettings->programSettings->matrixSize*i+j] = dis(gen);
            d->norma = (d->A[executionSettings->programSettings->matrixSize*i+j] > d->norma) ? d->A[executionSettings->programSettings->matrixSize*i+j] : d->norma;
        }
    }
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        d->b[i] = 0.0;
        d->ipvt[i] = i;
    }
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            d->b[j] += d->A[executionSettings->programSettings->matrixSize*j+i];
        }
    }
    return d;
}

bool  
linpack::LinpackBenchmark::validateOutputAndPrintError(linpack::LinpackData &data) {
    uint n= executionSettings->programSettings->matrixSize;
    auto newdata = generateInputData();
    for (int i = 0; i < n; i++) {
        newdata->b[i] = -newdata->b[i];
    }
    linpack::dmxpy(n, newdata->b, n, n, data.b, newdata->A);
    HOST_DATA_TYPE resid = 0.0;
    HOST_DATA_TYPE normx = 0.0;

    for (int i = 0; i < n; i++) {
        resid = (resid > fabs(newdata->b[i])) ? resid : fabs(newdata->b[i]);
        normx = (normx > fabs(data.b[i])) ? normx : fabs(data.b[i]);
    }

    HOST_DATA_TYPE eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    HOST_DATA_TYPE residn = resid / (n*newdata->norma*normx*eps);
    //std::cout << resid << ", " << norma << ", " << normx << std::endl;
    std::cout << "  norm. resid        resid       "\
                 "machep       x[0]-1     x[n-1]-1" << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << residn << std::setw(ENTRY_SPACE)
              << resid << std::setw(ENTRY_SPACE) << eps
              << std::setw(ENTRY_SPACE) << data.b[0]-1 << std::setw(ENTRY_SPACE)
              << data.b[n-1]-1 << std::endl;

    return residn < 100;
}

/**
Standard LU factorization on a block with fixed size

Case 1 of Zhangs description
*/
void
linpack::gefa_ref(HOST_DATA_TYPE* a, unsigned n, unsigned lda, cl_int* ipvt) {
    for (int i = 0; i < n; i++) {
        ipvt[i] = i;
    }
    // For each diagnonal element
    for (int k = 0; k < n - 1; k++) {
        HOST_DATA_TYPE max_val = fabs(a[k * lda + k]);
        int pvt_index = k;
        for (int i = k + 1; i < n; i++) {
            if (max_val < fabs(a[i * lda + k])) {
                pvt_index = i;
                max_val = fabs(a[i * lda + k]);
            }
        }

        for (int i = k; i < n; i++) {
            HOST_DATA_TYPE tmp_val = a[k * lda + i];
            a[k * lda + i] = a[pvt_index * lda + i];
            a[pvt_index * lda + i] = tmp_val;
        }
        ipvt[k] = pvt_index;

        // For each element below it
        for (int i = k + 1; i < n; i++) {
            a[i * lda + k] *= -1.0 / a[k * lda + k];
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[i * lda + j] += a[i * lda + k] * a[k * lda + j];
            }
        }

#ifdef DEBUG
        std::cout << "A(k=" << k <<"): " << std::endl;
                for (int i= 0; i < n; i++) {
                    for (int j=0; j < n; j++) {
                        std::cout << a[i*lda + j] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout <<  std::endl;
#endif

    }
}

void
linpack::gesl_ref(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, cl_int* ipvt, unsigned n, unsigned lda) {
    auto b_tmp = new HOST_DATA_TYPE[n];
#pragma omp parallel default(shared)
    {
#pragma omp for
        for (int k = 0; k < n; k++) {
            b_tmp[k] = b[k];
        }

        // solve l*y = b
        // For each row in matrix
#pragma omp single
        for (int k = 0; k < n - 1; k++) {
            if (ipvt[k] != k) {
                HOST_DATA_TYPE tmp = b_tmp[k];
                b_tmp[k] = b_tmp[ipvt[k]];
                b_tmp[ipvt[k]] = tmp;
            }
            // For each row below add
#pragma omp parallel for
            for (int i = k + 1; i < n; i++) {
                // add solved upper row to current row
                b_tmp[i] += b_tmp[k] * a[lda * i + k];
            }
        }

        // now solve  u*x = y
#pragma omp single
        for (int k = n - 1; k >= 0; k--) {
            b_tmp[k] = b_tmp[k] / a[lda * k + k];
#pragma omp parallel for
            for (int i = 0; i < k; i++) {
                b_tmp[i] -= b_tmp[k] * a[lda * i + k];
            }
        }
#pragma omp for
        for (int k = 0; k < n; k++) {
            b[k] = b_tmp[k];
        }
    }
    delete [] b_tmp;
}

void linpack::dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m) {
    for (int i=0; i < n1; i++) {
        for (int j=0; j < n2; j++) {
            y[i] = y[i] + x[j] * m[ldm*i + j];
        }
    }
}
