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

/* Related header files */
#include "random_access_functionality.hpp"

/* C++ standard library headers */
#include <iostream>
#include <string>
#include <limits>
#include <iomanip>
#include <memory>
#include <vector>

/* External library headers */
#include "CL/cl.hpp"
#include "cxxopts.hpp"

/* Project's headers */
#include "setup/fpga_setup.hpp"
#include "setup/common_benchmark_io.hpp"
#include "execution.h"

/**
Print the benchmark Results

@param results The result struct provided by the calculation call
@param dataSize The size of the used data array

*/
void
printResults(std::shared_ptr<bm_execution::ExecutionResults> results,
                  size_t dataSize, double error) {
    std::cout << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GUOPS"
              << std::setw(ENTRY_SPACE) << "error" << std::endl;

    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tmin = std::numeric_limits<double>::max();
    double gups = static_cast<double>(4 * dataSize) / 1000000000;
    for (double currentTime : results->times) {
        tmean +=  currentTime;
        if (currentTime < tmin) {
            tmin = currentTime;
        }
    }
    tmean = tmean / results->times.size();

    std::cout << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << gups / tmin
              << std::setw(ENTRY_SPACE) << (100.0 * error)
              << std::endl;
}

/**
 Generates the value of the random number after a desired number of updates

 @param n number of random number updates

 @return The random number after n number of updates
 */
HOST_DATA_TYPE
starts(HOST_DATA_TYPE_SIGNED n) {
    HOST_DATA_TYPE m2[BIT_SIZE];

    while (n < 0) {
        n += PERIOD;
    }
    while (n > 0) {
        n -= PERIOD;
    }

    if (n == 0) {
        return 1;
    }

    HOST_DATA_TYPE temp = 1;
    for (int i=0; i < BIT_SIZE; i++) {
        m2[i] = temp;
        for (int j=0; j < 2; j++) {
            HOST_DATA_TYPE_SIGNED v = 0;
            if (((HOST_DATA_TYPE_SIGNED) temp) < 0) {
                v = POLY;
            }
            temp = (temp << 1) ^ v;
        }
    }
    // DATA_TYPE i = BIT_SIZE - 2;
    // while (i >= 0 && !((n >> i) & 1)) {
    //     i--;
    // }
    int i = 0;
    for (i=BIT_SIZE - 2; i >= 0; i--) {
        if ((n >> i) & 1) {
            break;
        }
    }

    HOST_DATA_TYPE ran = 2;
    while (i > 0) {
        temp = 0;
        for (int j=0; j < BIT_SIZE; j++) {
            if ((ran >> j) & 1) {
                temp ^= m2[j];
            }
        }
        ran = temp;
        i--;
        if ((n >> i) & 1) {
            HOST_DATA_TYPE_SIGNED v = 0;
            if (((HOST_DATA_TYPE_SIGNED) ran) < 0) {
                v = POLY;
            }
            ran = (ran << 1) ^v;
        }
    }
    return ran;
}

double checkRandomAccessResults(HOST_DATA_TYPE* result_array, size_t array_size){
    HOST_DATA_TYPE temp = 1;
    for (HOST_DATA_TYPE i=0; i < 4L*array_size; i++) {
        HOST_DATA_TYPE_SIGNED v = 0;
        if (((HOST_DATA_TYPE_SIGNED)temp) < 0) {
            v = POLY;
        }
        temp = (temp << 1) ^ v;
        result_array[(temp >> 3) & (array_size - 1)] ^= temp;
    }

    double errors = 0;
#pragma omp parallel for reduction(+:errors)
    for (HOST_DATA_TYPE i=0; i< array_size; i++) {
        if (result_array[i] != i) {
            errors++;
        }
    }
    return errors / array_size;
}

void generateInputData(HOST_DATA_TYPE* data, size_t dataSize) {
    for (HOST_DATA_TYPE j=0; j < dataSize ; j++) {
        data[j] = j;
    }
}
