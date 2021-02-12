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
    matrixSize(results["s"].as<uint>()), isDiagonallyDominant(results.count("uniform") == 0) {

}

std::map<std::string, std::string>
linpack::LinpackProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        return map;
}

linpack::LinpackData::LinpackData(cl::Context context, size_t size) : norma(0.0), context(context) {
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
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("uniform", "Generate a uniform matrix instead of a diagonally dominant");
}

std::unique_ptr<linpack::LinpackExecutionTimings>
linpack::LinpackBenchmark::executeKernel(LinpackData &data) {
    return bm_execution::calculate(*executionSettings, data.A, data.b, data.ipvt);
}

void
linpack::LinpackBenchmark::collectAndPrintResults(const linpack::LinpackExecutionTimings &output) {
    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tlumean = 0;
    double tslmean = 0;
    double tmin = std::numeric_limits<double>::max();
    double lu_min = std::numeric_limits<double>::max();
    double sl_min = std::numeric_limits<double>::max();

    double gflops_lu = ((2.0e0*(static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)))/ 3.0) / 1.0e9; 
    double gflops_sl = (2.0*(static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)))/1.0e9;
    for (int i =0; i < output.gefaTimings.size(); i++) {
        double currentTime = output.gefaTimings[i] + output.geslTimings[i];
        tmean +=  currentTime;
        tlumean +=  output.gefaTimings[i];
        tslmean += output.geslTimings[i];
        if (currentTime < tmin) {
            tmin = currentTime;
        }
        if (output.gefaTimings[i] < lu_min) {
            lu_min = output.gefaTimings[i];
        }
        if (output.geslTimings[i] < sl_min) {
            sl_min = output.geslTimings[i];
        }
    }
    tmean = tmean / output.gefaTimings.size();
    tlumean = tlumean / output.gefaTimings.size();
    tslmean = tslmean / output.geslTimings.size();

     std::cout << std::setw(ENTRY_SPACE)
              << "Method" << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "total" << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << ((gflops_lu + gflops_sl) / tmin)
              << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "GEFA" << std::setw(ENTRY_SPACE)
            << lu_min << std::setw(ENTRY_SPACE) << tlumean
            << std::setw(ENTRY_SPACE) << ((gflops_lu) / lu_min)
            << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "GESL" << std::setw(ENTRY_SPACE)
              << sl_min << std::setw(ENTRY_SPACE) << tslmean
              << std::setw(ENTRY_SPACE) << (gflops_sl / sl_min)
              << std::endl;
}

std::unique_ptr<linpack::LinpackData>
linpack::LinpackBenchmark::generateInputData() {
    auto d = std::unique_ptr<linpack::LinpackData>(new linpack::LinpackData(*executionSettings->context ,executionSettings->programSettings->matrixSize));
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    d->norma = 0.0;
    if (executionSettings->programSettings->isDiagonallyDominant) {
        /*
        Generate a diagonally dominant matrix by using pseudo random number in the range (0,1)
        */
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            // initialize diagonal value
            d->A[executionSettings->programSettings->matrixSize*j+j] = 0;
            // fill a single column of the matrix
            for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
                if (i != j) {
                    HOST_DATA_TYPE temp = dis(gen);
                    d->A[executionSettings->programSettings->matrixSize*j+i] = temp;
                    // diagonal element of the current column will contain the sum of allother values in the column
                    d->A[executionSettings->programSettings->matrixSize*j+j] += temp;
                }
            }
            // the biggest value is automatically the diagnonal value
            d->norma = d->A[executionSettings->programSettings->matrixSize*j+j];
        }
    }
    else {
        /*
        Generate uniform matrix by using pseudo random number in the range (0,1)
        */
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            // fill a single column of the matrix
            for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
                HOST_DATA_TYPE temp = dis(gen);
                d->A[executionSettings->programSettings->matrixSize*i+j] = temp;
                d->norma = (temp > d->norma) ? temp : d->norma;
            }
        }
    }
    // initialize other vectors
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        d->b[i] = 0.0;
        d->ipvt[i] = i;
    }
    // Generate vector b by accumulating the rows of the matrix.
    // This will lead to a result vector x with ones on every position
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            d->b[j] += d->A[executionSettings->programSettings->matrixSize*i+j];
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
    linpack::dmxpy(n, newdata->b, n, n, data.b, newdata->A, false);
    HOST_DATA_TYPE resid = 0.0;
    HOST_DATA_TYPE normx = 0.0;

    for (int i = 0; i < n; i++) {
        resid = (resid > fabs(newdata->b[i])) ? resid : fabs(newdata->b[i]);
        normx = (normx > fabs(data.b[i])) ? normx : fabs(data.b[i]);
    }

    HOST_DATA_TYPE eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    HOST_DATA_TYPE residn = resid / (n*newdata->norma*normx*eps);

#ifndef NDEBUG
    if (residn > 1) {
        auto ref_result = generateInputData();
        // For each column right of current diagonal element
        for (int j = 0; j < n; j++) {
            // For each element below it
            for (int i = 0; i < n; i++) {
                std::cout << ref_result->A[n * j + i] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        // For each column right of current diagonal element
        for (int j = 0; j < n; j++) {
            // For each element below it
            for (int i = 0; i < n; i++) {
                std::cout << data.A[n * j + i] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        if (executionSettings->programSettings->isDiagonallyDominant) {
            linpack::gefa_ref_nopvt(ref_result->A, n, n);
            linpack::gesl_ref_nopvt(ref_result->A, ref_result->b, n, n);
        }
        else {
            linpack::gefa_ref(ref_result->A, n, n, ref_result->ipvt);
            linpack::gesl_ref(ref_result->A, ref_result->b, ref_result->ipvt, n, n);
        }
        // For each column right of current diagonal element
        for (int j = 0; j < n; j++) {
            // For each element below it
            for (int i = 0; i < n; i++) {
                std::cout << ref_result->A[n * j + i] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

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
            if (max_val < fabs(a[k * lda + i])) {
                pvt_index = i;
                max_val = fabs(a[k * lda + i]);
            }
        }

        for (int i = k; i < n; i++) {
            HOST_DATA_TYPE tmp_val = a[i * lda + k];
            a[i * lda + k] = a[i * lda + pvt_index];
            a[i * lda + pvt_index] = tmp_val;
        }
        ipvt[k] = pvt_index;

        // For each element below it
        for (int i = k + 1; i < n; i++) {
            a[k * lda + i] *= -1.0 / a[k * lda + k];
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[j * lda + i] += a[k * lda + i] * a[j * lda + k];
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
    {
        for (int k = 0; k < n; k++) {
            b_tmp[k] = b[k];
        }

        // solve l*y = b
        // For each row in matrix
        for (int k = 0; k < n - 1; k++) {
            if (ipvt[k] != k) {
                HOST_DATA_TYPE tmp = b_tmp[k];
                b_tmp[k] = b_tmp[ipvt[k]];
                b_tmp[ipvt[k]] = tmp;
            }
            // For each row below add
            for (int i = k + 1; i < n; i++) {
                // add solved upper row to current row
                b_tmp[i] += b_tmp[k] * a[lda * k + i];
            }
        }

        // now solve  u*x = y
        for (int k = n - 1; k >= 0; k--) {
            b_tmp[k] = b_tmp[k] / a[lda * k + k];
            for (int i = 0; i < k; i++) {
                b_tmp[i] -= b_tmp[k] * a[lda * k + i];
            }
        }
        for (int k = 0; k < n; k++) {
            b[k] = b_tmp[k];
        }
    }
    delete [] b_tmp;
}

void linpack::dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m, bool transposed) {
    for (int i=0; i < n1; i++) {
        for (int j=0; j < n2; j++) {
            y[i] = y[i] + x[j] * (transposed ? m[ldm*i + j] :m[ldm*j + i]);
        }
    }
}

void
linpack::gefa_ref_nopvt(HOST_DATA_TYPE* a, unsigned n, unsigned lda) {
    // For each diagnonal element
    for (int k = 0; k < n; k++) {
        // Store negatie invers of diagonal elements to get rid of some divisions afterwards!
        a[k * lda + k] = -1.0 / a[k * lda + k];
        // For each element below it
        for (int i = k + 1; i < n; i++) {
            a[k * lda + i] *= a[k * lda + k];
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[j * lda + i] += a[k * lda + i] * a[j * lda + k];
            }
        }

#ifdef DEBUG
        std::cout << "A(k=" << k << "): " << std::endl;
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
linpack::gesl_ref_nopvt(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, unsigned n, unsigned lda) {
    auto b_tmp = new HOST_DATA_TYPE[n];

        for (int k = 0; k < n; k++) {
            b_tmp[k] = b[k];
        }

        // solve l*y = b
        // For each row in matrix
        for (int k = 0; k < n - 1; k++) {
            // For each row below add
            for (int i = k + 1; i < n; i++) {
                // add solved upper row to current row
                b_tmp[i] += b_tmp[k] * a[lda * k + i];
            }
        }

        // now solve  u*x = y
        for (int k = n - 1; k >= 0; k--) {
            HOST_DATA_TYPE scale = b_tmp[k] * a[lda * k + k];
            b_tmp[k] = -scale;
            for (int i = 0; i < k; i++) {
                b_tmp[i] += scale * a[lda * k + i];
            }
        }
        for (int k = 0; k < n; k++) {
            b[k] = b_tmp[k];
        }
    delete [] b_tmp;
}
