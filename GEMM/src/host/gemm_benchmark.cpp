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

#include "gemm_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

gemm::GEMMProgramSettings::GEMMProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["b"].as<uint>() * results["m"].as<uint>()), blockSize(results["b"].as<uint>()) {

}

std::map<std::string, std::string>
gemm::GEMMProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        return map;
}

gemm::GEMMData::GEMMData(cl::Context context, uint size) : normtotal(0.0), alpha(0.5), beta(2.0), context(context) {
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    B = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    C = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    C_out = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&A), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C_out), 4096, size * size * sizeof(HOST_DATA_TYPE));
#endif
}

gemm::GEMMData::~GEMMData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void**>(A));
    clSVMFree(context(), reinterpret_cast<void**>(B));
    clSVMFree(context(), reinterpret_cast<void**>(C));
    clSVMFree(context(), reinterpret_cast<void**>(C_out));
#else
    free(A);
    free(B);
    free(C);
    free(C_out);
#endif
}

gemm::GEMMBenchmark::GEMMBenchmark(int argc, char* argv[]) {
    setupBenchmark(argc, argv);
}

gemm::GEMMBenchmark::GEMMBenchmark() {}

void
gemm::GEMMBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
            ("m", "Matrix size in number of blocks in a single dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
            ("b", "Block size in number of values in one dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(BLOCK_SIZE)));
}

std::unique_ptr<gemm::GEMMExecutionTimings>
gemm::GEMMBenchmark::executeKernel(GEMMData &data) {
    return bm_execution::calculate(*executionSettings, data.A, data.B, data.C, data.C_out, data.alpha, data.beta);
}

void
gemm::GEMMBenchmark::printResults(const gemm::GEMMExecutionTimings &output) {
    std::cout << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tmin = std::numeric_limits<double>::max();

    double gflops = 2.0 * (static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize)
                        *static_cast<double>(executionSettings->programSettings->matrixSize))/1.0e9;
    for (double currentTime : output.timings) {
        tmean +=  currentTime;
        if (currentTime < tmin) {
            tmin = currentTime;
        }
    }
    tmean = tmean / output.timings.size();

    std::cout << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << gflops / tmin
              << std::endl;

}

std::unique_ptr<gemm::GEMMData>
gemm::GEMMBenchmark::generateInputData() {
    auto d = std::unique_ptr<gemm::GEMMData>(new gemm::GEMMData(*executionSettings->context, executionSettings->programSettings->matrixSize));
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            d->A[executionSettings->programSettings->matrixSize*i+j] = dis(gen);
            d->B[executionSettings->programSettings->matrixSize*i+j] = dis(gen);
            d->C[executionSettings->programSettings->matrixSize*i+j] = dis(gen);
            d->C_out[executionSettings->programSettings->matrixSize*i+j] = 0.0;
            d->normtotal = std::max(std::max(d->normtotal, d->A[executionSettings->programSettings->matrixSize*i+j]), 
                                        std::max(d->B[executionSettings->programSettings->matrixSize*i+j], 
                                        d->C[executionSettings->programSettings->matrixSize*i+j]));
        }
    }
    return d;
}

bool  
gemm::GEMMBenchmark::validateOutputAndPrintError(gemm::GEMMData &data) {
    auto ref_data = generateInputData();

    gemm_ref(ref_data->A, ref_data->B, ref_data->C, executionSettings->programSettings->matrixSize, 0.5, 2.0);

    HOST_DATA_TYPE resid = 0.0;
    HOST_DATA_TYPE normx = 0.0;

    for (int i = 0; i < executionSettings->programSettings->matrixSize * executionSettings->programSettings->matrixSize; i++) {
        resid = (resid > fabs(data.C_out[i] - ref_data->C[i])) ? resid : fabs(data.C_out[i] - ref_data->C[i]);
        normx = (normx > fabs(data.C_out[i])) ? normx : fabs(data.C_out[i]);
    }

    HOST_DATA_TYPE eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    HOST_DATA_TYPE residn = resid / (executionSettings->programSettings->matrixSize*executionSettings->programSettings->matrixSize*ref_data->normtotal*normx*eps);

    std::cout << "  norm. resid        resid       "\
                 "machep" << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << residn << std::setw(ENTRY_SPACE)
              << resid << std::setw(ENTRY_SPACE) << eps
              << std::endl;

    return residn < 1.0;
}

void 
gemm::gemm_ref(HOST_DATA_TYPE* a,HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
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
