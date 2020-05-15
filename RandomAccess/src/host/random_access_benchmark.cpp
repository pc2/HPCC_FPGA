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

#include "random_access_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

random_access::RandomAccessProgramSettings::RandomAccessProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    dataSize(results["d"].as<size_t>()),
    kernelReplications(results["r"].as<uint>()) {

}

std::map<std::string, std::string>
random_access::RandomAccessProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        std::stringstream ss;
        ss << dataSize << " (" << static_cast<double>(dataSize * sizeof(HOST_DATA_TYPE)) << " Byte )";
        map["Array Size"] = ss.str();
        map["Kernel Replications"] = std::to_string(kernelReplications);
        return map;
}

random_access::RandomAccessBenchmark::RandomAccessBenchmark(int argc, char* argv[]) {
    setupBenchmark(argc, argv);
}

random_access::RandomAccessBenchmark::RandomAccessBenchmark() {}

void
random_access::RandomAccessBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("d", "Size of the data array",
            cxxopts::value<size_t>()->default_value(std::to_string(DEFAULT_ARRAY_LENGTH)))
        ("r", "Number of kernel replications used",
            cxxopts::value<uint>()->default_value(std::to_string(NUM_KERNEL_REPLICATIONS)))
        ("single-kernel", "Use the single kernel implementation");
}

std::shared_ptr<random_access::RandomAccessExecutionTimings>
random_access::RandomAccessBenchmark::executeKernel(const hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings> &settings, RandomAccessData &data) {
    return bm_execution::calculate(settings, data.data);
}

/**
Prints the execution results to stdout

@param results The execution results
*/
void
random_access::RandomAccessBenchmark::printResults(const hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings> &settings, const random_access::RandomAccessExecutionTimings &output) {
    std::cout << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GUOPS" << std::endl;

    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tmin = std::numeric_limits<double>::max();
    double gups = static_cast<double>(4 * settings.programSettings->dataSize) / 1000000000;
    for (double currentTime : output.times) {
        tmean +=  currentTime;
        if (currentTime < tmin) {
            tmin = currentTime;
        }
    }
    tmean = tmean / output.times.size();

    std::cout << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << gups / tmin
              << std::endl;

}

std::shared_ptr<random_access::RandomAccessData>
random_access::RandomAccessBenchmark::generateInputData(const hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings> &settings) {
    HOST_DATA_TYPE *data;
#ifdef USE_SVM
    data = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            settings.programSettings->dataSize * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&data), 4096, settings.programSettings->dataSize * sizeof(HOST_DATA_TYPE));
#endif

    for (HOST_DATA_TYPE j=0; j < settings.programSettings->dataSize ; j++) {
        data[j] = j;
    }

    return std::shared_ptr<RandomAccessData>(new RandomAccessData{data});
}

bool  
random_access::RandomAccessBenchmark::validateOutputAndPrintError(const hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings> &settings ,random_access::RandomAccessData &data, const random_access::RandomAccessExecutionTimings &output) {
    HOST_DATA_TYPE temp = 1;
    for (HOST_DATA_TYPE i=0; i < 4L*settings.programSettings->dataSize; i++) {
        HOST_DATA_TYPE_SIGNED v = 0;
        if (((HOST_DATA_TYPE_SIGNED)temp) < 0) {
            v = POLY;
        }
        temp = (temp << 1) ^ v;
        data.data[(temp >> 3) & (settings.programSettings->dataSize - 1)] ^= temp;
    }

    double errors = 0;
#pragma omp parallel for reduction(+:errors)
    for (HOST_DATA_TYPE i=0; i< settings.programSettings->dataSize; i++) {
        if (data.data[i] != i) {
            errors++;
        }
    }
    std::cout  << "Error: " << (static_cast<double>(errors) / settings.programSettings->dataSize) * 100 
                << "%" << std::endl;

    return (static_cast<double>(errors) / settings.programSettings->dataSize) < 0.01;
}