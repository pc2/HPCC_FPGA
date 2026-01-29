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

#ifndef SRC_HOST_TRANSPOSE_BENCHMARK_H_
#define SRC_HOST_TRANSPOSE_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "parameters.h"
#include "hpcc_benchmark.hpp"
#include "transpose_data.hpp"
#ifdef USE_OCL_HOST
#include "execution_types/execution_intel.hpp"
#include "execution_types/execution_intel_pq.hpp"
#include "execution_types/execution_pcie.hpp"
#include "execution_types/execution_pcie_pq.hpp"
#endif
#ifdef USE_XRT_HOST
#include "execution_types/execution_xrt_pcie_pq.hpp"
#ifdef USE_ACCL
#include "execution_types/execution_xrt_accl_pq.hpp"
#include "execution_types/execution_xrt_accl_stream_pq_sendrecv.hpp"
#include "execution_types/execution_xrt_accl_stream_pq.hpp"
#endif
#endif
#include "execution_types/execution_cpu.hpp"
#include "communication_types.hpp"

#include "data_handlers/data_handler_types.h"
#include "data_handlers/diagonal.hpp"
#include "data_handlers/pq.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {

/**
 * @brief Implementation of the transpose benchmark
 * 
 */
template<class TDevice, class TContext, class TProgram> 
class TransposeBenchmark : 
public hpcc_base::HpccFpgaBenchmark<TransposeProgramSettings,TDevice, TContext, TProgram, TransposeData<TContext>> {
protected:

    /**
     * @brief Additional input parameters of the transpose benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override {
        options.add_options()
            ("m", "Matrix size in number of blocks in one dimension",
                cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
            ("b", "Block size in number of values in one dimension",
                cxxopts::value<uint>()->default_value(std::to_string(BLOCK_SIZE)))
            ("p", "Value of P that equals the width of the PQ grid of FPGAs. Q is determined by the world size.",
                cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_P_VALUE)))
            ("distribute-buffers", "Distribute buffers over memory banks. This will use three memory banks instead of one for a single kernel replication, but kernel replications may interfere. This is an Intel only attribute, since buffer placement is decided at compile time for Xilinx FPGAs.")
            ("handler", "Specify the used data handler that distributes the data over devices and memory banks",
                cxxopts::value<std::string>()->default_value(DEFAULT_DIST_TYPE))
            ("copy-a", "Create a copy of matrix A for each kernel replication")
            ("accl-stream", "Use design with user kernels directly connected to CCLO");
    }

    std::unique_ptr<transpose::data_handler::TransposeDataHandler<TDevice, TContext, TProgram>> dataHandler;

public:

    /**
     * @brief Random access specific implementation of the data generation
     * 
     * @return std::unique_ptr<TransposeData> The input and output data of the benchmark
     */
    std::unique_ptr<TransposeData<TContext>>
    generateInputData() override {
        return this->dataHandler->generateData(*(this->executionSettings));
    }

    /**
     * @brief Set the data handler object by calling the function with the matching template argument
     * 
     */
    void
    setTransposeDataHandler(transpose::data_handler::DataHandlerType dataHandlerIdentifier) {
        switch (dataHandlerIdentifier) {
            case transpose::data_handler::DataHandlerType::diagonal: this->dataHandler = std::unique_ptr<transpose::data_handler::TransposeDataHandler<TDevice, TContext, TProgram>>(new transpose::data_handler::DistributedDiagonalTransposeDataHandler<TDevice, TContext, TProgram>(this->mpi_comm_rank, this->mpi_comm_size)); break;
            case transpose::data_handler::DataHandlerType::pq: this->dataHandler = std::unique_ptr<transpose::data_handler::TransposeDataHandler<TDevice, TContext, TProgram>>(new transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>(this->mpi_comm_rank, this->mpi_comm_size, this->executionSettings->programSettings->p)); break;
            default: throw std::runtime_error("Could not match selected data handler: " + transpose::data_handler::handlerToString(dataHandlerIdentifier));
        }
    }

    /**
     * @brief Transpose specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     */
    void
    executeKernel(TransposeData<TContext> &data) override {
        switch (this->executionSettings->programSettings->communicationType) {
#ifdef USE_OCL_HOST
            case hpcc_base::CommunicationType::intel_external_channels: 
                                    if (this->executionSettings->programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
                                        this->timings = transpose::fpga_execution::intel::calculate(*(this->executionSettings), data);
                                    }
                                    else {
                                        this->timings = transpose::fpga_execution::intel_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler));
                                    } break;
            case hpcc_base::CommunicationType::pcie_mpi :                                 
                                    if (this->executionSettings->programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
                                        this->timings = transpose::fpga_execution::pcie::calculate(*(this->executionSettings), data, *dataHandler);
                                    }
                                    else {
                                        this->timings = transpose::fpga_execution::pcie_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler));
                                    } break;
#endif
#ifdef USE_XRT_HOST
            case hpcc_base::CommunicationType::pcie_mpi:
                                    this->timings = transpose::fpga_execution::pcie_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler)); break;
#ifdef USE_ACCL
            // case hpcc_base::CommunicationType::accl:
            //                         return transpose::fpga_execution::accl_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler)); break;
            case hpcc_base::CommunicationType::accl: 
                                    if (this->executionSettings->programSettings->useAcclStreams) {
                                        auto h = reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler);
                                        if (!h.getP() == h.getQ()) {
                                            this->timings = transpose::fpga_execution::accl_stream_sendrecv_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler));
                                        }
                                        else {
                                            this->timings = transpose::fpga_execution::accl_stream_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler));
                                        }
                                    } else {
                                        this->timings = transpose::fpga_execution::accl_pq::calculate(*(this->executionSettings), data, reinterpret_cast<transpose::data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>&>(*this->dataHandler));
                                    } break;
#endif
#endif
#ifdef MKL_FOUND
            case hpcc_base::CommunicationType::cpu_only : this->timings = transpose::fpga_execution::cpu::calculate(*(this->executionSettings), data, *dataHandler); break;
#endif
            default: throw std::runtime_error("No calculate method implemented for communication type " + commToString(this->executionSettings->programSettings->communicationType));
        }
    }

    /**
     * @brief Transpose specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutput(TransposeData<TContext> &data) override {

        // exchange the data using MPI depending on the chosen distribution scheme
        this->dataHandler->exchangeData(data);

#ifndef NDEBUG
        std::vector<HOST_DATA_TYPE> oldA(this->executionSettings->programSettings->blockSize * this->executionSettings->programSettings->blockSize * data.numBlocks);
        std::copy(data.A, data.A + oldA.size(), oldA.data());
#endif

        this->dataHandler->reference_transpose(data);

        double max_error = 0.0;
        int error_count = 0;
        for (size_t i = 0; i < this->executionSettings->programSettings->blockSize * this->executionSettings->programSettings->blockSize * data.numBlocks; i++) {
            max_error = std::max(std::abs<double>(data.A[i]), max_error);
            if (std::abs<double>(data.A[i]) - 100 * std::numeric_limits<HOST_DATA_TYPE>::epsilon() > 0.0) {
                error_count++;
            }
        }

#ifndef NDEBUG
        // Only print debug info for PQ handlers (diagonal handlers don't have getHeightforRank/getWidthforRank)
        auto* pq_handler = dynamic_cast<data_handler::DistributedPQTransposeDataHandler<TDevice, TContext, TProgram>*>(this->dataHandler.get());
        if (pq_handler != nullptr && error_count > 0 && this->mpi_comm_rank == 0) {
            long height_per_rank = pq_handler->getHeightforRank();
            long width_per_rank = pq_handler->getWidthforRank();
            std::cout << "A:" << std::endl;
            for (size_t j = 0; j < height_per_rank * data.blockSize; j++) {
                for (size_t i = 0; i < width_per_rank * data.blockSize; i++) {
                    std::cout << oldA[j * width_per_rank * data.blockSize + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << "B:" << std::endl;
            for (size_t j = 0; j < height_per_rank * data.blockSize; j++) {
                for (size_t i = 0; i < width_per_rank * data.blockSize; i++) {
                    std::cout << data.B[j * width_per_rank * data.blockSize + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << "Transposed A:" << std::endl;
            for (size_t j = 0; j < height_per_rank * data.blockSize; j++) {
                for (size_t i = 0; i < width_per_rank * data.blockSize; i++) {
                    std::cout << data.A[j * width_per_rank * data.blockSize + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
#endif

        double global_max_error = 0;
        int global_error_count = 0;
        MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&error_count, &global_error_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        this->errors.emplace("epsilon", std::numeric_limits<HOST_DATA_TYPE>::epsilon());
        this->errors.emplace("max_error", global_max_error);

        return static_cast<double>(global_max_error) < 100 * std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    }

    /**
     * @brief Transpose specific impelmentation of the error printing
     *
     */
    void
    printError() override {
        if (this->mpi_comm_rank == 0) {
            std::cout << "Maximum error: " << this->errors.at("max_error") << " < " << 100 * this->errors.at("epsilon") <<  std::endl;
            std::cout << "Mach. Epsilon: " << this->errors.at("epsilon")  << std::endl;
        }
    }

    /**
     * @brief Transpose specific implementation of collecting the execution results
     * 
     */
    void
    collectResults() override {
        double flops = static_cast<double>(this->executionSettings->programSettings->matrixSize) * this->executionSettings->programSettings->matrixSize;

        // Number of experiment repetitions
        uint number_measurements = this->timings.at("calculation").size();
        std::vector<double> max_measures(number_measurements);
        std::vector<double> max_transfers(number_measurements);
#ifdef _USE_MPI_
            // Copy the object variable to a local variable to make it accessible to the lambda function
            int mpi_size = this->mpi_comm_size;
            MPI_Reduce(this->timings.at("calculation").data(), max_measures.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(this->timings.at("transfer").data(), max_transfers.data(), number_measurements, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
            std::copy(this->timings.at("calculation").begin(), this->timings.at("calculation").end(), max_measures.begin());
            std::copy(this->timings.at("transfer").begin(), this->timings.at("transfer").end(), max_transfers.begin());
#endif

        double avgCalculationTime = accumulate(max_measures.begin(), max_measures.end(), 0.0)
                                    / max_measures.size();
        this->results.emplace("avg_calc_t", hpcc_base::HpccResult(avgCalculationTime, "s"));

        double minCalculationTime = *min_element(max_measures.begin(), max_measures.end());
        this->results.emplace("min_calc_t", hpcc_base::HpccResult(minCalculationTime, "s"));

        double avgTransferTime = accumulate(max_transfers.begin(), max_transfers.end(), 0.0)
                                    / max_transfers.size();
        this->results.emplace("avg_transfer_t", hpcc_base::HpccResult(avgTransferTime, "s"));

        double minTransferTime = *min_element(max_transfers.begin(), max_transfers.end());
        this->results.emplace("min_transfer_t", hpcc_base::HpccResult(minTransferTime, "s"));
        
        this->results.emplace("avg_t", hpcc_base::HpccResult(avgCalculationTime + avgTransferTime, "s"));
        this->results.emplace("min_t", hpcc_base::HpccResult(minCalculationTime + minTransferTime, "s"));

        this->results.emplace("avg_calc_flops", hpcc_base::HpccResult(flops / avgCalculationTime * 1.0e-9, "GFLOP/s"));
        this->results.emplace("max_calc_flops", hpcc_base::HpccResult(flops / minCalculationTime * 1.0e-9, "GFLOP/s"));
        this->results.emplace("avg_mem_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / avgCalculationTime * 1.0e-9, "GB/s"));
        this->results.emplace("max_mem_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / minCalculationTime * 1.0e-9, "GB/s"));
        this->results.emplace("avg_transfer_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / avgTransferTime * 1.0e-9, "GB/s"));
        this->results.emplace("max_transfer_bandwidth", hpcc_base::HpccResult(flops * sizeof(HOST_DATA_TYPE) * 3 / minTransferTime * 1.0e-9, "GB/s"));
    }
    
    /**
     * @brief Transpose specific implementation of printing the execution results
     * 
     */
    void
    printResults() override {
        if (this->mpi_comm_rank == 0) {
            std::cout << std::setw(ENTRY_SPACE) << " "
                << std::left << std::setw(ENTRY_SPACE) << "total time"
                << std::setw(ENTRY_SPACE) << "transfer time"
                << std::setw(ENTRY_SPACE) << "calc time"
                << std::setw(ENTRY_SPACE) << "calc FLOPS"
                << std::setw(ENTRY_SPACE) << "Memory Bandwidth"
                << std::setw(ENTRY_SPACE) << "PCIe Bandwidth"
                << std::right << std::endl;
            std::cout << std::setw(ENTRY_SPACE) << "avg: "
                << this->results.at("avg_t")
                << this->results.at("avg_transfer_t")
                << this->results.at("avg_calc_t")
                << this->results.at("avg_calc_flops")
                << this->results.at("avg_mem_bandwidth")
                << this->results.at("avg_transfer_bandwidth")
                << std::endl;
            std::cout << std::setw(ENTRY_SPACE) << "best: " 
                << this->results.at("min_t")
                << this->results.at("min_transfer_t")
                << this->results.at("min_calc_t")
                << this->results.at("max_calc_flops")
                << this->results.at("max_mem_bandwidth")
                << this->results.at("max_transfer_bandwidth")
                << std::endl;
        }
    }

    /**
     * @brief Construct a new Transpose Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    TransposeBenchmark(int argc, char* argv[]) : hpcc_base::HpccFpgaBenchmark<transpose::TransposeProgramSettings,TDevice, TContext, TProgram, transpose::TransposeData<TContext>>(argc, argv) {
        if (this->setupBenchmark(argc, argv)) {
            this->setTransposeDataHandler(this->executionSettings->programSettings->dataHandlerIdentifier);
        }
    }

    /**
     * @brief Construct a new Transpose Benchmark object
     */
    TransposeBenchmark() : hpcc_base::HpccFpgaBenchmark<transpose::TransposeProgramSettings,TDevice, TContext, TProgram, transpose::TransposeData<TContext>>() {}

};

} // namespace transpose


#endif // SRC_HOST_TRANSPOSE_BENCHMARK_H_
