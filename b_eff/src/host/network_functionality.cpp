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
#include "parameters.h"

/**
Parses and returns program options using the cxxopts library.
Supports the following parameters:
    - file name of the FPGA kernel file (-f,--file)
    - number of repetitions (-n)
    - number of kernel replications (-r)
    - data size (-d)
    - use memory interleaving
@see https://github.com/jarro2783/cxxopts

@return program settings that are created from the given program arguments
*/
std::shared_ptr<ProgramSettings>
parseProgramParameters(int argc, char *argv[]) {
    // Defining and parsing program options
    cxxopts::Options options(argv[0], PROGRAM_DESCRIPTION);
    options.add_options()
            ("f,file", "Kernel file name", cxxopts::value<std::string>())
            ("n", "Number of repetitions",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_REPETITIONS)))
            ("l", "Inital looplength of Kernel",
             cxxopts::value<uint>()->default_value(std::to_string(1u << 15u)))
            ("device", "Index of the device that has to be used. If not given you "\
        "will be asked which device to use if there are multiple devices "\
        "available.", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_DEVICE)))
            ("platform", "Index of the platform that has to be used. If not given "\
        "you will be asked which platform to use if there are multiple "\
        "platforms available.",
             cxxopts::value<int>()->default_value(std::to_string(DEFAULT_PLATFORM)))
            ("h,help", "Print this help");
    cxxopts::ParseResult result = options.parse(argc, argv);
    
    if (result.count("h")) {
        // Just print help when argument is given
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Check parsed options and handle special cases
    if (result.count("f") <= 0) {
        // Path to the kernel file is mandatory - exit if not given!
        std::cerr << "Kernel file must be given! Aborting" << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }

    // Create program settings from program arguments
    std::shared_ptr<ProgramSettings> sharedSettings(
            new ProgramSettings{result["n"].as<uint>(),
                                result["l"].as<uint>(),
                                result["platform"].as<int>(),
                                result["device"].as<int>(),
                                result["f"].as<std::string>()});
    return sharedSettings;
}


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

/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings retrieved from the command line
 * @param device The device used for execution
 */
void printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                             const cl::Device &device) {// Give setup summary
    std::cout << PROGRAM_DESCRIPTION << std::endl << HLINE;
    std::cout << "Summary:" << std::endl
              << "Repetitions:         " << programSettings->numRepetitions
              << std::endl
              << "Kernel file:         " << programSettings->kernelFileName
              << std::endl;
    std::cout << "Device:              "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << HLINE
              << "Start benchmark using the given configuration." << std::endl
              << HLINE;
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