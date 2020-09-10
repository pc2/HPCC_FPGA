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
#include "execution.h"
#include "parameters.h"

network::NetworkProgramSettings::NetworkProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    looplength(results["l"].as<uint>()) {

}

std::map<std::string, std::string>
network::NetworkProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Loop Length"] = std::to_string(looplength);
        return map;
}

network::NetworkBenchmark::NetworkBenchmark(int argc, char* argv[]) {
    setupBenchmark(argc, argv);
}

network::NetworkBenchmark::NetworkBenchmark() {}

void
network::NetworkBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("l", "Inital looplength of Kernel",
             cxxopts::value<uint>()->default_value(std::to_string(1u << 15u)));
}

std::unique_ptr<network::NetworkExecutionTimings>
network::NetworkBenchmark::executeKernel(NetworkData &data) {
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<std::shared_ptr<network::ExecutionTimings>> timing_results;

    for (cl_uint size : data.messageSizes) {
        if (world_rank == 0) {
            std::cout << "Measure for " << size << " Byte" << std::endl;
        }
        cl_uint looplength = std::max((executionSettings->programSettings->looplength) / ((size + (CHANNEL_WIDTH)) / (CHANNEL_WIDTH)), 1u);
        timing_results.push_back(bm_execution::calculate(*executionSettings, size, looplength));
    }

    std::unique_ptr<network::NetworkExecutionTimings> collected_results = std::unique_ptr<network::NetworkExecutionTimings> (new network::NetworkExecutionTimings());
    if (world_rank > 0) {
        for (const auto& t : timing_results) {
            MPI_Send(&(t->messageSize),
                     1,
                     MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&(t->looplength),
                     1,
                     MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&(t->calculationTimings.front()),
                     executionSettings->programSettings->numRepetitions,
                     MPI_DOUBLE, 0,  2, MPI_COMM_WORLD);
        }
    } else {
        std::cout << "Collect results over MPI.";
        int k = 0;
        for (int size : data.messageSizes) {
            std::vector<std::shared_ptr<network::ExecutionTimings>> tmp_timings;
            std::cout << ".";
            for (int i=1; i < world_size; i++) {
                auto execution_result = std::shared_ptr<network::ExecutionTimings>( new network::ExecutionTimings {
                    0,0,std::vector<double>(executionSettings->programSettings->numRepetitions)
                });
                MPI_Status status;
                MPI_Recv(&(execution_result->messageSize),
                         1,
                         MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result->looplength),
                         1,
                         MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result->calculationTimings.front()),
                         executionSettings->programSettings->numRepetitions,
                         MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
                tmp_timings.push_back(execution_result);
                if (execution_result->messageSize != size) {
                    std::cerr << "Wrong message size: " << execution_result->messageSize << " != " << size << " from rank " << i << std::endl;
                    exit(2);
                }
            }
            tmp_timings.push_back(timing_results[k]);
            k++;
            collected_results->timings.emplace(size, std::make_shared<std::vector<std::shared_ptr<network::ExecutionTimings>>>(tmp_timings));
        }
        std::cout << " done!" << std::endl;
    }

        return collected_results;
}

void
network::NetworkBenchmark::collectAndPrintResults(const network::NetworkExecutionTimings &output) {
    std::vector<double> maxBandwidths;

    std::cout << std::setw(ENTRY_SPACE) << "MSize" << "   "
            << std::setw(ENTRY_SPACE) << "looplength" << "   "
            << std::setw(ENTRY_SPACE) << "transfer" << "   "
            << std::setw(ENTRY_SPACE) << "B/s" << std::endl;

    std::vector<double> totalMaxMinCalculationTime;
    for (int i =0; i < output.timings.size(); i++) {
        totalMaxMinCalculationTime.push_back(0.0);
    }
    int i = 0;
    for (const auto& msgSizeResults : output.timings) {
        for (const auto& r : *msgSizeResults.second) {
            double localMinCalculationTime = *min_element(r->calculationTimings.begin(), r->calculationTimings.end());
            totalMaxMinCalculationTime[i] = std::max(totalMaxMinCalculationTime[i], localMinCalculationTime);
        }
        i++;
    }
    i = 0;
    for (const auto& msgSizeResults : output.timings) {
        int looplength = msgSizeResults.second->at(0)->looplength;
        // The total sent data in bytes will be:
        // #Nodes * message_size * looplength * 2
        // the * 2 is because we have two kernels per bitstream that will send and receive simultaneously.
        // This will be divided by half of the maximum of the minimum measured runtime over all ranks.
        double maxCalcBW = static_cast<double>(msgSizeResults.second->size() * 2 * msgSizeResults.first * looplength)
                                                            / (totalMaxMinCalculationTime[i]);

        maxBandwidths.push_back(maxCalcBW);

        std::cout << std::setw(ENTRY_SPACE) << msgSizeResults.first << "   "
                  << std::setw(ENTRY_SPACE) << looplength << "   "
                  << std::setw(ENTRY_SPACE) << totalMaxMinCalculationTime[i] << "   "
                  << std::setw(ENTRY_SPACE)  << maxCalcBW
                  << std::endl;
        i++;
    }


    double b_eff = accumulate(maxBandwidths.begin(), maxBandwidths.end(), 0.0) / maxBandwidths.size();

    std::cout << std::endl << "b_eff = " << b_eff << " B/s" << std::endl;
}

std::unique_ptr<network::NetworkData>
network::NetworkBenchmark::generateInputData() {
    auto d = std::unique_ptr<network::NetworkData>(new network::NetworkData());
    for (uint i = 0; i < 13; i++) {
        d->messageSizes.push_back(1u << i);
    }
    cl_uint fourKB = 1u << 13u;
    for (uint i=1; i <= 8; i++) {
        d->messageSizes.push_back(fourKB * (1u << i));
    }
    return d;
}

bool  
network::NetworkBenchmark::validateOutputAndPrintError(network::NetworkData &data) {
    // TODO: No data returned from kernel to validate. Implement such a runtime validation!
    return true;
}

