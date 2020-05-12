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

#include "network_functionality.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "cxxopts.hpp"
#include "setup/fpga_setup.hpp"
#include "setup/common_benchmark_io.hpp"
#include "parameters.h"

/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(bm_execution::CollectedResultMap results) {

    std::vector<double> maxBandwidths;

    std::cout << std::setw(ENTRY_SPACE) << "MSize" << "   "
            << std::setw(ENTRY_SPACE) << "looplength" << "   "
            << std::setw(ENTRY_SPACE) << "transfer" << "   "
            << std::setw(ENTRY_SPACE) << "B/s" << std::endl;

    std::vector<double> totalMaxMinCalculationTime;
    for (int i =0; i < results.size(); i++) {
        totalMaxMinCalculationTime.push_back(0.0);
    }
    int i = 0;
    for (const auto& msgSizeResults : results) {
        for (const auto& r : *msgSizeResults.second) {
            double localMinCalculationTime = *min_element(r->calculationTimings.begin(), r->calculationTimings.end());
            totalMaxMinCalculationTime[i] = std::max(totalMaxMinCalculationTime[i], localMinCalculationTime);
        }
        i++;
    }
    i = 0;
    for (const auto& msgSizeResults : results) {
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

std::vector<cl_uint> getMessageSizes() {
    std::vector<cl_uint> sizes;
    for (uint i = 0; i < 13; i++) {
        sizes.push_back(1u << i);
    }
    cl_uint fourKB = 1u << 13u;
    for (uint i=1; i <= 8; i++) {
        sizes.push_back(fourKB * (1u << i));
    }
    return sizes;
}