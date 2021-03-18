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

transpose::TransposeBenchmark::TransposeBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
    setTransposeDataHandler(executionSettings->programSettings->dataHandlerIdentifier);
}

void
transpose::TransposeBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("m", "Matrix size in number of blocks in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("b", "Block size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(BLOCK_SIZE)))
        ("distribute-buffers", "Distribute buffers over memory banks. This will use three memory banks instead of one for a single kernel replication, but kernel replications may interfere. This is an Intel only attribute, since buffer placement is decided at compile time for Xilinx FPGAs.")
        ("handler", "Specify the used data handler that distributes the data over devices and memory banks",
            cxxopts::value<std::string>()->default_value(TRANSPOSE_HANDLERS_DIST_DIAG));
}

std::unique_ptr<transpose::TransposeExecutionTimings>
transpose::TransposeBenchmark::executeKernel(TransposeData &data) {
    return bm_execution::calculate(*executionSettings, data);
}

void
transpose::TransposeBenchmark::collectAndPrintResults(const transpose::TransposeExecutionTimings &output) {
    double flops = static_cast<double>(executionSettings->programSettings->matrixSize) * executionSettings->programSettings->matrixSize;

    // Number of experiment repetitions
    uint number_measurements = output.calculationTimings.size();
    std::vector<double> max_measures(number_measurements);
#ifdef _USE_MPI_
        // Copy the object variable to a local variable to make it accessible to the lambda function
        int mpi_size = mpi_comm_size;
        MPI_Reduce(output.calculationTimings.data(), max_measures.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
        std::copy(output.calculationTimings.begin(), output.calculationTimings.end(), max_measures.begin());
#endif

    double avgCalculationTime = accumulate(max_measures.begin(), max_measures.end(), 0.0)
                                / max_measures.size();
    double minCalculationTime = *min_element(max_measures.begin(), max_measures.end());

    double avgCalcFLOPS = flops / avgCalculationTime;
    double maxCalcFLOPS = flops / minCalculationTime;
    double avgMemBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / avgCalculationTime;
    double avgNetworkBandwidth = flops * sizeof(HOST_DATA_TYPE) / avgCalculationTime;
    double maxMemBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / minCalculationTime;
    double maxNetworkBandwidth = flops * sizeof(HOST_DATA_TYPE) / minCalculationTime;




    if (mpi_comm_rank == 0) {
        std::cout << "              calc    calc FLOPS    Net [GB/s]    Mem [GB/s]" << std::endl;
        std::cout << "avg:   " << avgCalculationTime
                << "   " << avgCalcFLOPS
                << "   " << avgNetworkBandwidth
                << "   " << avgMemBandwidth
                << std::endl;
        std::cout << "best:  " << minCalculationTime
                << "   " << maxCalcFLOPS
                << "   " << maxNetworkBandwidth
                << "   " << maxMemBandwidth
                << std::endl;
    }
}

std::unique_ptr<transpose::TransposeData>
transpose::TransposeBenchmark::generateInputData() {
    return dataHandler->generateData(*executionSettings);
}

bool  
transpose::TransposeBenchmark::validateOutputAndPrintError(transpose::TransposeData &data) {

    // exchange the data using MPI depending on the chosen distribution scheme
    dataHandler->exchangeData(data);

    int block_offset = executionSettings->programSettings->blockSize * executionSettings->programSettings->blockSize;
    for (int b = 0; b < data.numBlocks; b++) {
        for (int i = 0; i < executionSettings->programSettings->blockSize; i++) {
            for (int j = 0; j < executionSettings->programSettings->blockSize; j++) {
                data.A[b * block_offset + j * executionSettings->programSettings->blockSize + i] -= (data.result[b * block_offset + i * executionSettings->programSettings->blockSize + j] 
                                                                            - data.B[b * block_offset + i * executionSettings->programSettings->blockSize + j]);
            }
        }
    }

    double max_error = 0.0;
    for (int i = 0; i < executionSettings->programSettings->blockSize * executionSettings->programSettings->blockSize * data.numBlocks; i++) {
        max_error = std::max(fabs(data.A[i]), max_error);
    }

    double global_max_error = 0;
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_comm_rank == 0) {
        std::cout << "Maximum error: " << global_max_error << " < " << 100 * std::numeric_limits<HOST_DATA_TYPE>::epsilon() <<  std::endl;
        std::cout << "Mach. Epsilon: " << std::numeric_limits<HOST_DATA_TYPE>::epsilon() << std::endl;
    }

    return static_cast<double>(global_max_error) < 100 * std::numeric_limits<HOST_DATA_TYPE>::epsilon();
}

void
transpose::TransposeBenchmark::setTransposeDataHandler(std::string dataHandlerIdentifier) {
    if (transpose::dataHandlerIdentifierMap.find(dataHandlerIdentifier) == transpose::dataHandlerIdentifierMap.end()) {
        throw std::runtime_error("Could not match selected data handler: " + dataHandlerIdentifier);
    }
    dataHandler = transpose::dataHandlerIdentifierMap[dataHandlerIdentifier](mpi_comm_rank, mpi_comm_size);
}
