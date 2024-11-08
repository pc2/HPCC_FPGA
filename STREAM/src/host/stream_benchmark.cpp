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

#include "stream_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.hpp"
#include "parameters.h"

stream::StreamProgramSettings::StreamProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    streamArraySize(results["s"].as<uint>()),
    useSingleKernel(!static_cast<bool>(results.count("multi-kernel"))) {

}

std::map<std::string, std::string>
stream::StreamProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Data Type"] = STR(HOST_DATA_TYPE);
        std::stringstream ss;
        ss << streamArraySize << " (" << static_cast<double>(streamArraySize * sizeof(HOST_DATA_TYPE)) << " Byte )";
        map["Array Size"] = ss.str();
        map["Kernel Type"] = (useSingleKernel ? "Single" : "Separate");
        return map;
}

stream::StreamData::StreamData(const cl::Context& _context, size_t size) : context(_context) {
#ifdef INTEL_FPGA
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            size * sizeof(HOST_DATA_TYPE), 1024));
    B = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            size * sizeof(HOST_DATA_TYPE), 1024));
    C = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            size * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&A), 64, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 64, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 64, size * sizeof(HOST_DATA_TYPE));
#endif
#endif
#ifdef XILINX_FPGA
    posix_memalign(reinterpret_cast<void**>(&A), 4096, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 4096, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 4096, size * sizeof(HOST_DATA_TYPE));
#endif
}

stream::StreamData::~StreamData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void*>(A));
    clSVMFree(context(), reinterpret_cast<void*>(B));
    clSVMFree(context(), reinterpret_cast<void*>(C));
#else
    free(A);
    free(B);
    free(C);
#endif
}

stream::StreamBenchmark::StreamBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
}

void
stream::StreamBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
        options.add_options()
            ("s", "Size of the data arrays",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_ARRAY_LENGTH)))
            ("multi-kernel", "Use the legacy multi kernel implementation");
}

void
stream::StreamBenchmark::executeKernel(StreamData &data) {
    timings = bm_execution::calculate(*executionSettings,
              data.A,
              data.B,
              data.C);
}

void
stream::StreamBenchmark::collectResults() {
    std::map<std::string,std::vector<double>> totalTimingsMap;
    for (auto v : timings) {
        // Number of experiment repetitions
        uint number_measurements = v.second.size();
        // create a new 
        std::vector<double> avg_measures(number_measurements);
#ifdef _USE_MPI_
        // Copy the object variable to a local variable to make it accessible to the lambda function
        int mpi_size = mpi_comm_size;
        MPI_Reduce(v.second.data(), avg_measures.data(), number_measurements, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        std::for_each(avg_measures.begin(),avg_measures.end(), [mpi_size](double& x) {x /= mpi_size;});
#else
        std::copy(v.second.begin(), v.second.end(), avg_measures.begin());
#endif

        double minTime = *min_element(v.second.begin(), v.second.end());
        double avgTime = accumulate(v.second.begin(), v.second.end(), 0.0)
                        / v.second.size();
        double maxTime = *max_element(v.second.begin(), v.second.end());

        double bestRate = (static_cast<double>(sizeof(HOST_DATA_TYPE)) * executionSettings->programSettings->streamArraySize * bm_execution::multiplicatorMap[v.first] / minTime) * 1.0e-6 * mpi_comm_size;
        
        results.emplace(v.first + "_min_t", hpcc_base::HpccResult(minTime, "s"));
        results.emplace(v.first + "_avg_t", hpcc_base::HpccResult(avgTime, "s"));
        results.emplace(v.first + "_max_t", hpcc_base::HpccResult(maxTime, "s"));
        results.emplace(v.first + "_best_rate", hpcc_base::HpccResult(bestRate, "MB/s"));
    }
}

void
stream::StreamBenchmark::printResults() {
    if (mpi_comm_rank == 0) {
        std::cout << std::left << std::setw(ENTRY_SPACE) << "Function";
        std::cout << std::setw(ENTRY_SPACE) << "Best Rate";
        std::cout << std::setw(ENTRY_SPACE) << "Avg time";
        std::cout << std::setw(ENTRY_SPACE) << "Min time" ;
        std::cout << std::setw(ENTRY_SPACE) << "Max time" << std::right << std::endl;

        for (auto key : keys) {
            std::cout << std::left << std::setw(ENTRY_SPACE) << key
                << results.at(key + "_best_rate")
                << results.at(key + "_avg_t")
                << results.at(key + "_min_t")
                << results.at(key + "_max_t")
                << std::right << std::endl;
        }
    }
}

std::unique_ptr<stream::StreamData>
stream::StreamBenchmark::generateInputData() {
    auto d = std::unique_ptr<stream::StreamData>(new StreamData(*executionSettings->context, executionSettings->programSettings->streamArraySize));
    for (int i=0; i< executionSettings->programSettings->streamArraySize; i++) {
        d->A[i] = 1.0;
        d->B[i] = 2.0;
        d->C[i] = 0.0;
    }

    return d;
}

bool  
stream::StreamBenchmark::validateOutput(stream::StreamData &data) {
    HOST_DATA_TYPE aj,bj,cj,scalar;
    double aSumErr,bSumErr,cSumErr;
    double aAvgErr,bAvgErr,cAvgErr;
    double epsilon;
    ssize_t	j;
    int	k,ierr,err;

    /* reproduce initialization */
    aj = static_cast<HOST_DATA_TYPE>(1.0);
    bj = static_cast<HOST_DATA_TYPE>(2.0);
    cj = static_cast<HOST_DATA_TYPE>(0.0);
    /* a[] is modified during timing check */
    aj = static_cast<HOST_DATA_TYPE>(2.0) * aj;
    /* now execute timing loop */
    scalar = static_cast<HOST_DATA_TYPE>(3.0);
    for (k=0; k<executionSettings->programSettings->numRepetitions; k++)
    {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
    for (j=0; j< executionSettings->programSettings->streamArraySize; j++) {
        aSumErr += std::abs(data.A[j] - aj);
        bSumErr += std::abs(data.B[j] - bj);
        cSumErr += std::abs(data.C[j] - cj);
    }
    aAvgErr = aSumErr / executionSettings->programSettings->streamArraySize;
    bAvgErr = bSumErr / executionSettings->programSettings->streamArraySize;
    cAvgErr = cSumErr / executionSettings->programSettings->streamArraySize;

#ifdef _USE_MPI_
    double totalAAvgErr = 0.0;
    double totalBAvgErr = 0.0;
    double totalCAvgErr = 0.0;
    MPI_Reduce(&aAvgErr, &totalAAvgErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bAvgErr, &totalAAvgErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cAvgErr, &totalAAvgErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    aAvgErr = totalAAvgErr / mpi_comm_size;
    bAvgErr = totalBAvgErr / mpi_comm_size;
    bAvgErr = totalBAvgErr / mpi_comm_size;
#endif

    bool success = true;
    if (mpi_comm_rank == 0) {
        errors.emplace("a_expected", aj);
        errors.emplace("a_average_error", aAvgErr);
        errors.emplace("a_average_relative_error", abs(aAvgErr)/aj);

        errors.emplace("b_expected", bj);
        errors.emplace("b_average_error", bAvgErr);
        errors.emplace("b_average_relative_error", abs(bAvgErr)/bj);

        errors.emplace("c_expected", cj);
        errors.emplace("c_average_error", cAvgErr);
        errors.emplace("c_average_relative_error", abs(cAvgErr)/cj);

        epsilon = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
        errors.emplace("epsilon", epsilon);

        if (abs(aAvgErr/aj) > epsilon) {
            success = false;
            ierr = 0;
            for (j=0; j<executionSettings->programSettings->streamArraySize; j++) {
                if (abs(data.A[j]/aj-1.0) > epsilon) {
                    ierr++;
                }
            }
            errors.emplace("a_error_count", ierr);
            ierr = 0;
        }
        if (abs(bAvgErr/bj) > epsilon) {
            success = false;
            ierr = 0;
            for (j=0; j<executionSettings->programSettings->streamArraySize; j++) {
                if (abs(data.B[j]/bj-1.0) > epsilon) {
                    ierr++;
                }
            }
            errors.emplace("b_error_count", ierr);
        }
        if (abs(cAvgErr/cj) > epsilon) {
            success = false;
            ierr = 0;
            for (j=0; j<executionSettings->programSettings->streamArraySize; j++) {
                if (abs(data.C[j]/cj-1.0) > epsilon) {
                    ierr++;
                }
            }
            errors.emplace("b_error_count", ierr);
        }
    }
    return success;
}

void
stream::StreamBenchmark::printError() {
    if (mpi_comm_rank == 0) {
        int err = 0;
        double epsilon = errors.at("epsilon");
        if (errors.at("a_average_relative_error") > epsilon) {
            err++;
            printf("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n", errors.at("epsilon"));
            printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", errors.at("a_expected"), errors.at("a_average_error"), errors.at("a_average_relative_error"));
            printf("     For array a[], %d errors were found.\n", errors.at("a_error_count"));
        }

        if (errors.at("b_average_relative_error") > epsilon) {
            err++;
            printf("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n", errors.at("epsilon"));
            printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", errors.at("b_expected"), errors.at("b_average_error"), errors.at("b_average_relative_error"));
            printf("     AvgRelAbsErr > Epsilon (%e)\n", errors.at("epsilon"));
            printf("     For array b[], %d errors were found.\n", errors.at("b_error_count"));
        }
        if (errors.at("c_average_relative_error") > epsilon) {
            err++;
            printf("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n", errors.at("epsilon"));
            printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", errors.at("c_expected"), errors.at("c_average_error"), errors.at("c_average_relative_error"));
            printf("     AvgRelAbsErr > Epsilon (%e)\n", errors.at("epsilon"));
            printf("     For array c[], %d errors were found.\n", errors.at("c_error_count"));
        }
        if (err == 0) {
            printf ("Solution Validates: avg error less than %e on all three arrays\n", errors.at("epsilon"));
        }
    }
}
