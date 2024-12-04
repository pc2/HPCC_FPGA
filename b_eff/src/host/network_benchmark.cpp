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

#include "network_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution_types/execution.hpp"
#include "parameters.h"

network::NetworkProgramSettings::NetworkProgramSettings(cxxopts::ParseResult &results)
    : hpcc_base::BaseSettings(results), maxLoopLength(results["u"].as<uint>()), minLoopLength(results["l"].as<uint>()),
      maxMessageSize(results["m"].as<uint>()), minMessageSize(results["min-size"].as<uint>()),
      stepSize(results["step-size"].as<uint>()), llOffset(results["o"].as<uint>()), llDecrease(results["d"].as<uint>()),
      pcie_reverse_write_pcie(results["pcie-read"].count()), pcie_reverse_read_pcie(results["pcie-write"].count()),
      pcie_reverse_execute_kernel(results["kernel-latency"].count()), pcie_reverse_batch(results["pcie-batch"].count()),
      pcie_reverse(results["pcie-reverse"].count())
#ifdef USE_XRT_HOST
      ,
      payload_size(results["payload-size"].as<uint>())
#endif
#ifdef USE_ACCL
      ,
      accl_from_programable_logic(results["accl-pl"].count()), accl_axi_stream(results["accl-stream"].count())
#endif
{

    pcie_reverse = pcie_reverse_execute_kernel | pcie_reverse_read_pcie | pcie_reverse_write_pcie;
}

std::map<std::string, std::string> network::NetworkProgramSettings::getSettingsMap()
{
    auto map = hpcc_base::BaseSettings::getSettingsMap();
    map["Loop Length"] = std::to_string(minLoopLength) + " - " + std::to_string(maxLoopLength);
    map["Message Sizes"] = "2^" + std::to_string(minMessageSize) + " - 2^" + std::to_string(maxMessageSize) + " Bytes";
    map["Max. UDP Payload"] = std::to_string((1 << payload_size) * 64) + " Bytes";
    return map;
}

network::NetworkData::NetworkDataItem::NetworkDataItem(unsigned int _messageSize, unsigned int _loopLength,
                                                       unsigned int replications)
    : messageSize(_messageSize), loopLength(_loopLength), validationBuffer((1 << _messageSize) * replications, 0)
{
    // TODO: fix the validation buffer size to use the variable number of kernel replications and channels
    // Validation data buffer should be big enough to fit the data of two channels
    // for every repetition. The number of kernel replications is fixed to 2, which
    // also needs to be multiplied with the buffer size
}

network::NetworkData::NetworkData(unsigned int max_looplength, unsigned int min_looplength,
                                  unsigned int min_messagesize, unsigned int max_messagesize, unsigned int stepsize,
                                  unsigned int offset, unsigned int decrease, unsigned int replications)
{
    uint decreasePerStep = (max_looplength - min_looplength) / decrease;
    for (uint i = min_messagesize; i <= max_messagesize; i += stepsize) {
        uint messageSizeDivOffset = (i > offset) ? i - offset : 0u;
        uint newLooplength = (max_looplength > messageSizeDivOffset * decreasePerStep)
                                 ? max_looplength - messageSizeDivOffset * decreasePerStep
                                 : 0u;
        uint looplength = std::max(newLooplength, min_looplength);
        this->items.push_back(NetworkDataItem(i, looplength, replications));
    }
}

network::NetworkBenchmark::NetworkBenchmark(int argc, char *argv[]) { setupBenchmark(argc, argv); }

network::NetworkBenchmark::NetworkBenchmark() {}

void network::NetworkBenchmark::addAdditionalParseOptions(cxxopts::Options &options)
{
    options.add_options()("u,upper", "Maximum number of repetitions per data size",
                          cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MAX_LOOP_LENGTH)))(
        "l,lower", "Minimum number of repetitions per data size",
        cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MIN_LOOP_LENGTH)))(
        "min-size", "Minimum Message Size", cxxopts::value<uint>()->default_value(std::to_string(0)))(
        "m", "Maximum message size", cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MAX_MESSAGE_SIZE)))(
        "step-size", "Step size to generate message sizes in the specified range",
        cxxopts::value<uint>()->default_value(std::to_string(1)))(
        "o", "Offset used before reducing repetitions",
        cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_LOOP_LENGTH_OFFSET)))(
        "d", "Number os steps the repetitions are decreased to its minimum",
        cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_LOOP_LENGTH_DECREASE)))
#ifdef USE_ACCL
        ("accl-pl", "Use second ACCL command kernel to schedule sends and recevs from PL")(
            "accl-stream", "Send and receive data to AXI streams instead of global memory")
#endif
#ifdef USE_XRT_HOST
            ("payload-size", "Maximum payload size used in 64 Byte chunks. Specified as powerr of two",
             cxxopts::value<uint>()->default_value(std::to_string(VNX_MAX_FRAME_SIZE)))
#endif
                ("pcie-read", "Use reverse PCIe experiment and measure PCIe read performance from device")(
                    "pcie-write", "Use reverse PCIe experiment and measure PCIe write performance from device")(
                    "kernel-latency", "Use reverse PCIe experiment and measure kernel execution latency")(
                    "pcie-batch",
                    "Execute the reverse PCIe experiments in batch mode to make use of the queues of the schedulers")(
                    "pcie-reverse", "Execute the reverse PCIe experiments");
}

void network::NetworkBenchmark::executeKernel(NetworkData &data)
{
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<network::ExecutionTimings> timing_results;

    for (auto &run : data.items) {
        if (world_rank == 0) {
            std::cout << std::dec << "Measure for " << (1 << run.messageSize) << " Byte" << std::endl;
        }
        network::ExecutionTimings timing;
        switch (executionSettings->programSettings->communicationType) {
        case hpcc_base::CommunicationType::cpu_only:
            timing = execution_types::cpu::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                     run.validationBuffer);
            break;
#ifndef USE_XRT_HOST
        case hpcc_base::CommunicationType::pcie_mpi:
            if (executionSettings->programSettings->pcie_reverse) {
                timing = execution_types::pcie_reverse::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                                  run.validationBuffer);
            } else {
                timing = execution_types::pcie::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                          run.validationBuffer);
            }
            break;
#ifdef INTEL_FPGA
        case hpcc_base::CommunicationType::intel_external_channels:
            timing = execution_types::iec::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                     run.validationBuffer);
            break;
#endif
#else
        case hpcc_base::CommunicationType::udp:
            timing = execution_types::udp::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                     run.validationBuffer);
            break;
#ifdef USE_ACCL
        case hpcc_base::CommunicationType::accl:
            if (!executionSettings->programSettings->accl_from_programable_logic) {
                if (!executionSettings->programSettings->accl_axi_stream) {
                    timing = execution_types::accl::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                              run.validationBuffer);
                } else {
                    timing = execution_types::accl_stream::calculate(*executionSettings, run.messageSize,
                                                                     run.loopLength, run.validationBuffer);
                }
            } else {
                if (!executionSettings->programSettings->accl_axi_stream) {
                    timing = execution_types::accl_pl::calculate(*executionSettings, run.messageSize, run.loopLength,
                                                                 run.validationBuffer);
                } else {
                    timing = execution_types::accl_pl_stream::calculate(*executionSettings, run.messageSize,
                                                                        run.loopLength, run.validationBuffer);
                }
            }
            break;
#endif
#endif
        default:
            throw std::runtime_error("Selected Communication type not supported: " +
                                     hpcc_base::commToString(executionSettings->programSettings->communicationType));
        }
        timing_results.push_back(timing);
    }

    if (world_rank > 0) {
        for (const auto &t : timing_results) {
            MPI_Send(&(t.messageSize), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&(t.looplength), 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&(t.calculationTimings.front()), executionSettings->programSettings->numRepetitions, MPI_DOUBLE, 0,
                     2, MPI_COMM_WORLD);
        }
    } else {
        std::cout << "Collect results over MPI.";
        int k = 0;
        for (auto &run : data.items) {
            std::vector<network::ExecutionTimings> tmp_timings;
            std::cout << ".";
            for (int i = 1; i < world_size; i++) {
                auto execution_result = network::ExecutionTimings{
                    0, 0, std::vector<double>(executionSettings->programSettings->numRepetitions)};
                MPI_Status status;
                MPI_Recv(&(execution_result.messageSize), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result.looplength), 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result.calculationTimings.front()),
                         executionSettings->programSettings->numRepetitions, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
                tmp_timings.push_back(execution_result);
                if (execution_result.messageSize != run.messageSize) {
                    std::cerr << "Wrong message size: " << execution_result.messageSize << " != " << run.messageSize
                              << " from rank " << i << std::endl;
                    throw std::runtime_error(
                        "Wrong message size received! Something went wrong in the MPI communication");
                }
            }
            tmp_timings.push_back(timing_results[k]);
            k++;
            collected_timings.emplace(run.messageSize, network::ExecutionResult{tmp_timings, 0.0, 0.0});
        }
        std::cout << " done!" << std::endl;
    }
    return;
}

void network::NetworkBenchmark::collectResults()
{
    std::vector<double> maxBandwidths;

    if (mpi_comm_rank == 0) {
        int i = 0;
        for (auto &timing : collected_timings) {
            for (auto &r : timing.second.execution_timings) {
                double localMinCalculationTime = *min_element(r.calculationTimings.begin(), r.calculationTimings.end());
                timing.second.maxMinCalculationTime =
                    std::max(timing.second.maxMinCalculationTime, localMinCalculationTime);
            }
            i++;
        }
        i = 0;
        for (auto &timing : collected_timings) {
            int looplength = timing.second.execution_timings.at(0).looplength;
            int messageSize = timing.first;
            int num_timings = timing.second.execution_timings.size();
            // The total sent data in bytes will be:
            // #Nodes * message_size * looplength * kernel_replications
            // the * kernel_replications is because we have multiple replications per bitstream that will send and
            // receive simultaneously. This will be divided by half of the maximum of the minimum measured runtime over
            // all ranks.
            timing.second.maxCalcBW = static_cast<double>(num_timings) *
                                      executionSettings->programSettings->kernelReplications * (1 << messageSize) *
                                      looplength / timing.second.maxMinCalculationTime;

            maxBandwidths.push_back(timing.second.maxCalcBW);

            i++;
        }

        results.emplace("b_eff", hpcc_base::HpccResult(accumulate(maxBandwidths.begin(), maxBandwidths.end(), 0.0) /
                                                           static_cast<double>(maxBandwidths.size()),
                                                       "B/s"));
    }
}

void network::NetworkBenchmark::printResults()
{
    std::cout << std::setw(ENTRY_SPACE) << "MSize" << "   " << std::setw(ENTRY_SPACE) << "looplength" << "   "
              << std::setw(ENTRY_SPACE) << "time [s]" << "   " << std::setw(ENTRY_SPACE) << "B/s" << std::endl;

    for (const auto &timing : collected_timings) {
        std::cout << std::setw(ENTRY_SPACE) << (1 << timing.first) << "   " << std::setw(ENTRY_SPACE)
                  << timing.second.execution_timings.at(0).looplength << "   " << std::setw(ENTRY_SPACE)
                  << timing.second.maxMinCalculationTime << "   " << std::setw(ENTRY_SPACE) << timing.second.maxCalcBW
                  << std::endl;
    }

    std::cout << std::endl << "b_eff = " << results.at("b_eff") << std::endl;
}

std::unique_ptr<network::NetworkData> network::NetworkBenchmark::generateInputData()
{
    // sanity check of input variables
    if (executionSettings->programSettings->minLoopLength > executionSettings->programSettings->maxLoopLength) {
        std::cerr << "WARNING: Loop Length: Minimum is bigger than maximum. Setting minimum to value of maximum."
                  << std::endl;
        executionSettings->programSettings->minLoopLength = executionSettings->programSettings->maxLoopLength;
    }
    if (executionSettings->programSettings->minMessageSize > executionSettings->programSettings->maxMessageSize) {
        std::cerr << "WARNING: Message Sizes: Minimum is bigger than maximum. Setting minimum to value of maximum."
                  << std::endl;
        executionSettings->programSettings->minMessageSize = executionSettings->programSettings->maxMessageSize;
    }
    auto d = std::unique_ptr<network::NetworkData>(new network::NetworkData(
        executionSettings->programSettings->maxLoopLength, executionSettings->programSettings->minLoopLength,
        executionSettings->programSettings->minMessageSize, executionSettings->programSettings->maxMessageSize,
        executionSettings->programSettings->stepSize, executionSettings->programSettings->llOffset,
        executionSettings->programSettings->llDecrease, executionSettings->programSettings->kernelReplications));
    return d;
}

bool network::NetworkBenchmark::validateOutput(network::NetworkData &data)
{
    unsigned total_error = 0;

    // For every data size in the data set
    for (const auto &item : data.items) {
        // check if the validation buffer contains the expected data
        HOST_DATA_TYPE expected_value = static_cast<HOST_DATA_TYPE>(item.messageSize & 255u);
        unsigned error_count = 0;
        HOST_DATA_TYPE failing_entry = 0;
        for (const auto &v : item.validationBuffer) {
            if (v != expected_value) {
                error_count++;
                failing_entry = v;
            }
        }
        if (error_count > 0) {
            errors.emplace(std::to_string(item.messageSize), error_count);
        }
        total_error += error_count;
    }

    // success only, if no error occured
    return total_error == 0;
}

void network::NetworkBenchmark::printError()
{
    for (const auto &error : errors) {
        std::cerr << "Validation data invalid for message size " << (1 << stoi(error.first)) << " in "
                  << int(error.second) << " cases!" << std::endl;
    }
}
