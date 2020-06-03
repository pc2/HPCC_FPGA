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

#include "transpose_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()) {

}

std::map<std::string, std::string>
transpose::TransposeProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        return map;
}

transpose::TransposeData::TransposeData(cl::Context context, uint size) : context(context) {
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    B = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    result = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void **>(&A), 64,
                sizeof(HOST_DATA_TYPE) * size * size);
    posix_memalign(reinterpret_cast<void **>(&B), 64,
                sizeof(HOST_DATA_TYPE) * size * size);
    posix_memalign(reinterpret_cast<void **>(&result), 64,
                sizeof(HOST_DATA_TYPE) * size * size);
#endif
}

transpose::TransposeData::~TransposeData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void*>(A));
    clSVMFree(context(), reinterpret_cast<void*>(B));
    clSVMFree(context(), reinterpret_cast<void*>(result));
#else
    free(A);
    free(B);
    free(result);
#endif
}

transpose::TransposeBenchmark::TransposeBenchmark(int argc, char* argv[]) {
    setupBenchmark(argc, argv);
}

transpose::TransposeBenchmark::TransposeBenchmark() {}

void
transpose::TransposeBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("m", "Matrix size in number of blocks in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("b", "Block size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(BLOCK_SIZE)));
}

std::unique_ptr<transpose::TransposeExecutionTimings>
transpose::TransposeBenchmark::executeKernel(TransposeData &data) {
    return bm_execution::calculate(*executionSettings, data.A, data.B, data.result);
}

void
transpose::TransposeBenchmark::printResults(const transpose::TransposeExecutionTimings &output) {
    double flops = executionSettings->programSettings->matrixSize * executionSettings->programSettings->matrixSize;

    double avgTransferTime = accumulate(output.transferTimings.begin(), output.transferTimings.end(), 0.0)
                             / output.transferTimings.size();
    double minTransferTime = *min_element(output.transferTimings.begin(), output.transferTimings.end());


    double avgCalculationTime = accumulate(output.calculationTimings.begin(), output.calculationTimings.end(), 0.0)
                                / output.calculationTimings.size();
    double minCalculationTime = *min_element(output.calculationTimings.begin(), output.calculationTimings.end());

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

std::unique_ptr<transpose::TransposeData>
transpose::TransposeBenchmark::generateInputData() {
    auto d = std::unique_ptr<transpose::TransposeData>(new transpose::TransposeData(*executionSettings->context, executionSettings->programSettings->matrixSize));

    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            d->A[i * executionSettings->programSettings->matrixSize + j] = dis(gen);
            d->B[j * executionSettings->programSettings->matrixSize + i] = dis(gen);
            d->result[j * executionSettings->programSettings->matrixSize + i] = 0.0;
        }
    }

    return d;
}

bool  
transpose::TransposeBenchmark::validateOutputAndPrintError(transpose::TransposeData &data) {
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            data.A[j * executionSettings->programSettings->matrixSize + i] -= data.result[i * executionSettings->programSettings->matrixSize + j] 
                                                                        - data.B[i * executionSettings->programSettings->matrixSize + j];
        }
    }

    double max_error = 0.0;
    for (int i = 0; i < executionSettings->programSettings->matrixSize * executionSettings->programSettings->matrixSize; i++) {
        max_error = std::max(fabs(data.A[i]), max_error);
    }

    std::cout << "Maximum error: " << max_error << std::endl;

    return (static_cast<double>(max_error) / executionSettings->programSettings->matrixSize) < 1.0e-6;
}