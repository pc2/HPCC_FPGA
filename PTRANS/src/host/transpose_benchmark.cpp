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
#include "fpga_execution/execution_intel.hpp"
#include "fpga_execution/execution_pcie.hpp"
#include "fpga_execution/execution_cpu.hpp"
#include "fpga_execution/communication_types.h"
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
        ("connectivity", "Specify the connectivity for the used bitstream", cxxopts::value<std::string>()->default_value(transpose::fpga_execution::comm_to_str_map.begin()->first))
        ("distribute-buffers", "Distribute buffers over memory banks. This will use three memory banks instead of one for a single kernel replication, but kernel replications may interfere. This is an Intel only attribute, since buffer placement is decided at compile time for Xilinx FPGAs.")
        ("handler", "Specify the used data handler that distributes the data over devices and memory banks",
            cxxopts::value<std::string>()->default_value(TRANSPOSE_HANDLERS_DIST_DIAG));
}

std::unique_ptr<transpose::TransposeExecutionTimings>
transpose::TransposeBenchmark::executeKernel(TransposeData &data) {
    switch (executionSettings->programSettings->communicationType) {
        case transpose::fpga_execution::CommunicationType::intel_external_channels: return transpose::fpga_execution::intel::calculate(*executionSettings, data); break;
        case transpose::fpga_execution::CommunicationType::pcie_mpi : return transpose::fpga_execution::pcie::calculate(*executionSettings, data, *dataHandler); break;
#ifdef MKL_FOUND
        case transpose::fpga_execution::CommunicationType::cpu_only : return transpose::fpga_execution::cpu::calculate(*executionSettings, data, *dataHandler); break;
#endif
        default: throw new std::runtime_error("No calculate method implemented for communication type " + transpose::fpga_execution::commToString(executionSettings->programSettings->communicationType));
    }
}

void
transpose::TransposeBenchmark::collectAndPrintResults(const transpose::TransposeExecutionTimings &output) {
    double flops = static_cast<double>(executionSettings->programSettings->matrixSize) * executionSettings->programSettings->matrixSize;

    // Number of experiment repetitions
    uint number_measurements = output.calculationTimings.size();
    std::vector<double> max_measures(number_measurements);
    std::vector<double> max_transfers(number_measurements);
#ifdef _USE_MPI_
        // Copy the object variable to a local variable to make it accessible to the lambda function
        int mpi_size = mpi_comm_size;
        MPI_Reduce(output.calculationTimings.data(), max_measures.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(output.transferTimings.data(), max_transfers.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
        std::copy(output.calculationTimings.begin(), output.calculationTimings.end(), max_measures.begin());
        std::copy(output.transferTimings.begin(), output.transferTimings.end(), max_transfers.begin());
#endif

    double avgCalculationTime = accumulate(max_measures.begin(), max_measures.end(), 0.0)
                                / max_measures.size();
    double minCalculationTime = *min_element(max_measures.begin(), max_measures.end());

    double avgTransferTime = accumulate(max_transfers.begin(), max_transfers.end(), 0.0)
                                / max_transfers.size();
    double minTransferTime = *min_element(max_transfers.begin(), max_transfers.end());

    double avgCalcFLOPS = flops / avgCalculationTime;
    double maxCalcFLOPS = flops / minCalculationTime;
    double avgMemBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / avgCalculationTime;
    double maxMemBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / minCalculationTime;
    double avgTransferBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / avgTransferTime;
    double maxTransferBandwidth = flops * sizeof(HOST_DATA_TYPE) * 3 / minTransferTime;




    if (mpi_comm_rank == 0) {
        std::cout << "       total [s]     transfer [s]  calc [s]      calc FLOPS    Mem [B/s]     PCIe [B/s]" << std::endl;
        std::cout << "avg:   " << (avgTransferTime + avgCalculationTime)
                << "   " << avgTransferTime
                << "   " << avgCalculationTime
                << "   " << avgCalcFLOPS
                << "   " << avgMemBandwidth
                << "   " << avgTransferBandwidth
                << std::endl;
        std::cout << "best:  " << (minTransferTime + minCalculationTime)
                << "   " << minTransferTime
                << "   " << minCalculationTime
                << "   " << maxCalcFLOPS
                << "   " << maxMemBandwidth
                << "   " << maxTransferBandwidth
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

    size_t block_offset = executionSettings->programSettings->blockSize * executionSettings->programSettings->blockSize;
    for (size_t b = 0; b < data.numBlocks; b++) {
        for (size_t i = 0; i < executionSettings->programSettings->blockSize; i++) {
            for (size_t j = 0; j < executionSettings->programSettings->blockSize; j++) {
                data.A[b * block_offset + j * executionSettings->programSettings->blockSize + i] -= (data.result[b * block_offset + i * executionSettings->programSettings->blockSize + j] 
                                                                            - data.B[b * block_offset + i * executionSettings->programSettings->blockSize + j]);
            }
        }
    }

    double max_error = 0.0;
    for (size_t i = 0; i < executionSettings->programSettings->blockSize * executionSettings->programSettings->blockSize * data.numBlocks; i++) {
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
