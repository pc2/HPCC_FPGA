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

#include "stream_functionality.hpp"

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
            ("s", "Size of the data arrays",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_ARRAY_LENGTH)))
            ("r", "Number of kernel replications used",
             cxxopts::value<uint>()->default_value(std::to_string(NUM_KERNEL_REPLICATIONS)))
#ifdef INTEL_FPGA
            ("i", "Use memory Interleaving")
#endif
            ("single-kernel", "Use the single kernel implementation")
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
                                result["s"].as<uint>(),
                                result["r"].as<uint>(),
#ifdef INTEL_FPGA
                                static_cast<bool>(result.count("i")),
#else
                                false,
#endif
                                result["platform"].as<int>(),
                                result["device"].as<int>(),
                                result["f"].as<std::string>(),
                                static_cast<bool>(result.count("single-kernel"))});
    return sharedSettings;
}


/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(std::shared_ptr<bm_execution::ExecutionTimings> results) {

    std::cout << std::setw(ENTRY_SPACE) << "Function";
    std::cout << std::setw(ENTRY_SPACE) << "Best Rate MB/s";
    std::cout << std::setw(ENTRY_SPACE) << "Avg time s";
    std::cout << std::setw(ENTRY_SPACE) << "Min time" ;
    std::cout << std::setw(ENTRY_SPACE) << "Max time" << std::endl;

    for (auto v : results->timings) {
        double minTime = *min_element(v.second.begin(), v.second.end());
        double avgTime = accumulate(v.second.begin(), v.second.end(), 0.0)
                         / v.second.size();
        double maxTime = *max_element(v.second.begin(), v.second.end());

        std::cout << std::setw(ENTRY_SPACE) << v.first;
        std::cout << std::setw(ENTRY_SPACE)
        << (static_cast<double>(sizeof(HOST_DATA_TYPE)) * results->arraySize * bm_execution::multiplicatorMap[v.first] / minTime) * 1.0e-6
                << std::setw(ENTRY_SPACE) << avgTime
                << std::setw(ENTRY_SPACE) << minTime
                << std::setw(ENTRY_SPACE) << maxTime << std::endl;
    }

}

/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings retrieved from the command line
 * @param device The device used for execution
 */
void printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                             const cl::Device &device) {// Give setup summary
    std::cout << PROGRAM_DESCRIPTION << std::endl;
    std::cout << "Version: " << VERSION << std::endl << HLINE;
    std::cout << "Summary:" << std::endl
              << "Array Size:          "
              << static_cast<double>(programSettings->streamArraySize * sizeof(HOST_DATA_TYPE)) << " Byte"
              << std::endl
              << "Data Type            " << STR(HOST_DATA_TYPE)
              << std::endl
              << "Repetitions:         " << programSettings->numRepetitions
              << std::endl
              << "Kernel replications: " << programSettings->kernelReplications
              << std::endl
              << "Kernel type:         " << (programSettings->useSingleKernel ? "Single" : "Separate")
              << std::endl
              << "Kernel file:         " << programSettings->kernelFileName
              << std::endl;
    std::cout << "Device:              "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << HLINE
              << "Start benchmark using the given configuration." << std::endl
              << HLINE;
}


void generateInputData(HOST_DATA_TYPE* A, HOST_DATA_TYPE* B, HOST_DATA_TYPE* C, unsigned array_size) {
    for (int i=0; i< array_size; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }
}

double checkSTREAMResult(const HOST_DATA_TYPE* A, const HOST_DATA_TYPE* B, const HOST_DATA_TYPE* C, unsigned repetitions,
        unsigned array_size) {
    HOST_DATA_TYPE aj,bj,cj,scalar;
    HOST_DATA_TYPE aSumErr,bSumErr,cSumErr;
    HOST_DATA_TYPE aAvgErr,bAvgErr,cAvgErr;
    double epsilon;
    ssize_t	j;
    int	k,ierr,err;

    /* reproduce initialization */
    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    /* a[] is modified during timing check */
    aj = 2.0E0 * aj;
    /* now execute timing loop */
    scalar = 3.0;
    for (k=0; k<repetitions; k++)
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
    for (j=0; j< array_size; j++) {
        aSumErr += abs(A[j] - aj);
        bSumErr += abs(B[j] - bj);
        cSumErr += abs(C[j] - cj);
        // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
    }
    aAvgErr = aSumErr / (HOST_DATA_TYPE) array_size;
    bAvgErr = bSumErr / (HOST_DATA_TYPE) array_size;
    cAvgErr = cSumErr / (HOST_DATA_TYPE) array_size;

    if (sizeof(HOST_DATA_TYPE) == 4) {
        epsilon = 1.e-6;
    }
    else if (sizeof(HOST_DATA_TYPE) == 8) {
        epsilon = 1.e-13;
    }
    else {
        printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(array_size));
        epsilon = 1.e-6;
    }

    err = 0;
    if (abs(aAvgErr/aj) > epsilon) {
        err++;
        printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
        ierr = 0;
        for (j=0; j<array_size; j++) {
            if (abs(A[j]/aj-1.0) > epsilon) {
                ierr++;
            }
        }
        printf("     For array a[], %d errors were found.\n",ierr);
    }
    if (abs(bAvgErr/bj) > epsilon) {
        err++;
        printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
        printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
        ierr = 0;
        for (j=0; j<array_size; j++) {
            if (abs(B[j]/bj-1.0) > epsilon) {
                ierr++;
            }
        }
        printf("     For array b[], %d errors were found.\n",ierr);
    }
    if (abs(cAvgErr/cj) > epsilon) {
        err++;
        printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
        printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
        printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
        ierr = 0;
        for (j=0; j<array_size; j++) {
            if (abs(C[j]/cj-1.0) > epsilon) {
                ierr++;
            }
        }
        printf("     For array c[], %d errors were found.\n",ierr);
    }
    if (err == 0) {
        printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
    }
    return err;
}