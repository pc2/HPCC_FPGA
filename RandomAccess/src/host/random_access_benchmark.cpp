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
    int mpi_size = 1;
#ifdef _USE_MPI_
     MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    auto map = hpcc_base::BaseSettings::getSettingsMap();
    std::stringstream ss;
    ss << dataSize << " (" << static_cast<double>(dataSize * sizeof(HOST_DATA_TYPE) * mpi_size) << " Byte )";
    map["Array Size"] = ss.str();
    map["Kernel Replications"] = std::to_string(kernelReplications);
    return map;
}

random_access::RandomAccessData::RandomAccessData(cl::Context& context, size_t size) : context(context) {
#ifdef USE_SVM
    data = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&data), 4096, size * sizeof(HOST_DATA_TYPE));
#endif
}

random_access::RandomAccessData::~RandomAccessData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void*>(data));
#else
    free(data);
#endif
}

random_access::RandomAccessBenchmark::RandomAccessBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
}

void
random_access::RandomAccessBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("d", "Size of the data array",
            cxxopts::value<size_t>()->default_value(std::to_string(DEFAULT_ARRAY_LENGTH)));
}

std::unique_ptr<random_access::RandomAccessExecutionTimings>
random_access::RandomAccessBenchmark::executeKernel(RandomAccessData &data) {
    return bm_execution::calculate(*executionSettings, data.data, mpi_comm_rank, mpi_comm_size);
}

void
random_access::RandomAccessBenchmark::collectAndPrintResults(const random_access::RandomAccessExecutionTimings &output) {

    std::vector<double> avgTimings(output.times.size());
#ifdef _USE_MPI_
    // Copy the object variable to a local variable to make it accessible to the lambda function
    int mpi_size = mpi_comm_size;
    MPI_Reduce(output.times.data(),avgTimings.data(),output.times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::for_each(avgTimings.begin(),avgTimings.end(), [mpi_size](double& x) {x /= mpi_size;});
#else
    std::copy(output.times.begin(), output.times.end(), avgTimings.begin());
#endif
    if (mpi_comm_rank == 0) {
        std::cout << std::setw(ENTRY_SPACE)
                << "best" << std::setw(ENTRY_SPACE) << "mean"
                << std::setw(ENTRY_SPACE) << "GUOPS" << std::endl;

        // Calculate performance for kernel execution
        double tmean = 0;
        double tmin = std::numeric_limits<double>::max();
        double gups = static_cast<double>(4 * executionSettings->programSettings->dataSize * mpi_comm_size) / 1000000000;
        for (double currentTime : avgTimings) {
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
}

bool
random_access::RandomAccessBenchmark::checkInputParameters() {
    bool validationResult = true;
    if ((mpi_comm_size == 0) || (mpi_comm_size & (mpi_comm_size - 1))) {
        // Number of MPI ranks is not a power of 2
        // This is not allowed, since the arithmetric on the input data works only on powers of 2
        std::cerr << "ERROR: Number of MPI ranks is " << mpi_comm_size << " which is not a power of two!" << std::endl;
        validationResult = false;
    }
    int data_per_replication = executionSettings->programSettings->dataSize / executionSettings->programSettings->kernelReplications;
    if ((data_per_replication == 0) || (data_per_replication & (data_per_replication - 1))) {
        std::cerr << "ERROR: Data chunk size for each kernel replication is not a power of 2!" << std::endl;
        validationResult = false;
    }
    return validationResult;
}

std::unique_ptr<random_access::RandomAccessData>
random_access::RandomAccessBenchmark::generateInputData() {
    auto d = std::unique_ptr<RandomAccessData>(new RandomAccessData(*executionSettings->context, executionSettings->programSettings->dataSize));
    for (HOST_DATA_TYPE j=0; j < executionSettings->programSettings->dataSize ; j++) {
        d->data[j] = mpi_comm_rank * executionSettings->programSettings->dataSize + j;
    }
    return d;
}

bool  
random_access::RandomAccessBenchmark::validateOutputAndPrintError(random_access::RandomAccessData &data) {

    HOST_DATA_TYPE* rawdata;
    if (mpi_comm_size > 1) {
#ifdef _USE_MPI_
        if (mpi_comm_rank == 0) {
            rawdata = new HOST_DATA_TYPE[executionSettings->programSettings->dataSize * mpi_comm_size];
        }
        MPI_Gather(data.data, executionSettings->programSettings->dataSize, MPI_LONG, 
                rawdata, executionSettings->programSettings->dataSize, MPI_LONG, 0, MPI_COMM_WORLD);
#endif
    }
    else {
        rawdata = data.data;
    }


    if (mpi_comm_rank == 0) {

        // Serially execute all pseudo random updates again
        // This should lead to the initial values in the data array, because XOR is a involutory function
        HOST_DATA_TYPE temp = 1;
        for (HOST_DATA_TYPE i=0; i < 4L*executionSettings->programSettings->dataSize * mpi_comm_size; i++) {
            HOST_DATA_TYPE_SIGNED v = 0;
            if (((HOST_DATA_TYPE_SIGNED)temp) < 0) {
                v = POLY;
            }
            temp = (temp << 1) ^ v;
            rawdata[(temp >> 3) & (executionSettings->programSettings->dataSize * mpi_comm_size - 1)] ^= temp;
        }

        double errors = 0;
#pragma omp parallel for reduction(+:errors)
        for (HOST_DATA_TYPE i=0; i< executionSettings->programSettings->dataSize * mpi_comm_size; i++) {
            if (rawdata[i] != i) {
                // If the array at index i does not contain i, it differs from the initial value and is counted as an error
                errors++;
            }
        }

        // The overall error is calculated in percent of the overall array size
        double error_ratio = static_cast<double>(errors) / (executionSettings->programSettings->dataSize * mpi_comm_size);
        std::cout  << "Error: " << error_ratio * 100 
                    << "%" << std::endl;

#ifdef _USE_MPI_
        if (mpi_comm_rank == 0 && mpi_comm_size > 1) {
            delete [] rawdata;
        }
#endif

        return error_ratio < 0.01;
    }

    // All other ranks skip validation and always return true
    return true;
}