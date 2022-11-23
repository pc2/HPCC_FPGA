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
#include "execution_types/execution_intel.hpp"
#include "execution_types/execution_intel_pq.hpp"
#include "execution_types/execution_pcie.hpp"
#include "execution_types/execution_pcie_pq.hpp"
#include "execution_types/execution_cpu.hpp"
#include "communication_types.hpp"

#include "data_handlers/data_handler_types.h"
#include "data_handlers/diagonal.hpp"
#include "data_handlers/pq.hpp"

#include "parameters.h"


transpose::TransposeBenchmark::TransposeBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    if (setupBenchmark(argc, argv)) {
        setTransposeDataHandler(executionSettings->programSettings->dataHandlerIdentifier);
    }
}

void
transpose::TransposeBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("m", "Matrix size in number of blocks in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("b", "Block size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(BLOCK_SIZE)))
        ("p", "Value of P that equals the width of the PQ grid of FPGAs. Q is determined by the world size.",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_P_VALUE)))
        ("distribute-buffers", "Distribute buffers over memory banks. This will use three memory banks instead of one for a single kernel replication, but kernel replications may interfere. This is an Intel only attribute, since buffer placement is decided at compile time for Xilinx FPGAs.")
        ("handler", "Specify the used data handler that distributes the data over devices and memory banks",
            cxxopts::value<std::string>()->default_value(DEFAULT_DIST_TYPE));
}

void
transpose::TransposeBenchmark::executeKernel(TransposeData &data) {
    switch (executionSettings->programSettings->communicationType) {
        case hpcc_base::CommunicationType::intel_external_channels: 
                                if (executionSettings->programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
                                    timings = transpose::fpga_execution::intel::calculate(*executionSettings, data);
                                }
                                else {
                                    timings = transpose::fpga_execution::intel_pq::calculate(*executionSettings, data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler&>(*dataHandler));
                                } break;
        case hpcc_base::CommunicationType::pcie_mpi :                                 
                                if (executionSettings->programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
                                    timings = transpose::fpga_execution::pcie::calculate(*executionSettings, data, *dataHandler);
                                }
                                else {
                                    timings = transpose::fpga_execution::pcie_pq::calculate(*executionSettings, data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler&>(*dataHandler));
                                } break;
#ifdef MKL_FOUND
        case hpcc_base::CommunicationType::cpu_only : return transpose::fpga_execution::cpu::calculate(*executionSettings, data, *dataHandler); break;
#endif
        default: throw std::runtime_error("No calculate method implemented for communication type " + commToString(executionSettings->programSettings->communicationType));
    }
}

void
transpose::TransposeBenchmark::collectResults() {
    double flops = static_cast<double>(executionSettings->programSettings->matrixSize) * executionSettings->programSettings->matrixSize;

    // Number of experiment repetitions
    uint number_measurements = timings.at("calculation").size();
    std::vector<double> max_measures(number_measurements);
    std::vector<double> max_transfers(number_measurements);
#ifdef _USE_MPI_
        // Copy the object variable to a local variable to make it accessible to the lambda function
        int mpi_size = mpi_comm_size;
        MPI_Reduce(timings.at("calculation").data(), max_measures.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(timings.at("transfer").data(), max_transfers.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
        std::copy(timings.at("calculation").begin(), timings.at("calculation").end(), max_measures.begin());
        std::copy(timings.at("transfer").begin(), timings.at("transfer").end(), max_transfers.begin());
#endif

    double avgCalculationTime = accumulate(max_measures.begin(), max_measures.end(), 0.0)
                                / max_measures.size();
    results.emplace("avg_calc_t", hpcc_base::HpccResult(avgCalculationTime, "s"));

    double minCalculationTime = *min_element(max_measures.begin(), max_measures.end());
    results.emplace("min_calc_t", hpcc_base::HpccResult(minCalculationTime, "s"));

    double avgTransferTime = accumulate(max_transfers.begin(), max_transfers.end(), 0.0)
                                / max_transfers.size();
    results.emplace("avg_transfer_t", hpcc_base::HpccResult(avgTransferTime, "s"));

    double minTransferTime = *min_element(max_transfers.begin(), max_transfers.end());
    results.emplace("min_transfer_t", hpcc_base::HpccResult(minTransferTime, "s"));
    
    results.emplace("avg_t", hpcc_base::HpccResult(avgCalculationTime + avgTransferTime, "s"));
    results.emplace("min_t", hpcc_base::HpccResult(minCalculationTime + minTransferTime, "s"));

    results.emplace("avg_calc_flops", hpcc_base::HpccResult(flops / avgCalculationTime, "GFLOP/s"));
    results.emplace("max_calc_flops", hpcc_base::HpccResult(flops / minCalculationTime, "GFLOP/s"));
    results.emplace("avg_mem_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / avgCalculationTime, "GB/s"));
    results.emplace("max_mem_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / minCalculationTime, "GB/s"));
    results.emplace("avg_transfer_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / avgTransferTime, "GB/s"));
    results.emplace("max_transfer_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / minTransferTime, "GB/s"));
}

void
transpose::TransposeBenchmark::printResults() {
    std::cout << "       total [s]     transfer [s]  calc [s]      calc FLOPS    Mem [B/s]     PCIe [B/s]" << std::endl;
    std::cout << "avg:   " << results.at("avg_t")
            << "   " << results.at("avg_transfer_t")
            << "   " << results.at("avg_calc_t")
            << "   " << results.at("avg_calc_flops")
            << "   " << results.at("avg_mem_bandwidth")
            << "   " << results.at("avg_transfer_bandwidth")
            << std::endl;
    std::cout << "best:  " << results.at("min_t")
            << "   " << results.at("min_transfer_t")
            << "   " << results.at("min_calc_t")
            << "   " << results.at("max_calc_flops")
            << "   " << results.at("max_mem_bandwidth")
            << "   " << results.at("max_transfer_bandwidth")
            << std::endl;
}

std::unique_ptr<transpose::TransposeData>
transpose::TransposeBenchmark::generateInputData() {
return dataHandler->generateData(*executionSettings);
}

bool  
transpose::TransposeBenchmark::validateOutputAndPrintError(transpose::TransposeData &data) {

    // exchange the data using MPI depending on the chosen distribution scheme
    dataHandler->exchangeData(data);

    dataHandler->reference_transpose(data);

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
transpose::TransposeBenchmark::setTransposeDataHandler(transpose::data_handler::DataHandlerType dataHandlerIdentifier) {
    switch (dataHandlerIdentifier) {
        case transpose::data_handler::DataHandlerType::diagonal: dataHandler = std::unique_ptr<transpose::data_handler::TransposeDataHandler>(new transpose::data_handler::DistributedDiagonalTransposeDataHandler(mpi_comm_rank, mpi_comm_size)); break;
        case transpose::data_handler::DataHandlerType::pq: dataHandler = std::unique_ptr<transpose::data_handler::TransposeDataHandler>(new transpose::data_handler::DistributedPQTransposeDataHandler(mpi_comm_rank, mpi_comm_size, executionSettings->programSettings->p)); break;
        default: throw std::runtime_error("Could not match selected data handler: " + transpose::data_handler::handlerToString(dataHandlerIdentifier));
    }
        

}
