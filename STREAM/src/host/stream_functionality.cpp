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
#include "setup/fpga_setup.hpp"
#include "parameters.h"
#include "program_settings.h"
#include "setup/common_benchmark_io.hpp"

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