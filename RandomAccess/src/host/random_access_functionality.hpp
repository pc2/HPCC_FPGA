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

#ifndef SRC_HOST_RANDOM_ACCESS_FUNCTIONALITY_H_
#define SRC_HOST_RANDOM_ACCESS_FUNCTIONALITY_H_

/* C++ standard library headers */
#include <memory>

/* Project's headers */
#include "execution.h"
#include "setup/fpga_setup.hpp"
#include "parameters.h"

/**
Prefix of the function name of the used kernel.
It will be used to construct the full function name for the case of replications.
The full name will be
*/
#define RANDOM_ACCESS_KERNEL "accessMemory_"

/**
Constants used to verify benchmark results
*/
#define POLY 7
#define PERIOD 1317624576693539401L

#define BIT_SIZE (sizeof(HOST_DATA_TYPE) * 8)

/**
 Generates the value of the random number after a desired number of updates

 @param n number of random number updates

 @return The random number after n number of updates
 */
HOST_DATA_TYPE
starts(HOST_DATA_TYPE_SIGNED n);

/**
Prints the execution results to stdout

@param results The execution results
@param dataSize Size of the used data array. Needed to calculate GUOP/s from
                timings
*/
void printResults(std::shared_ptr<bm_execution::ExecutionResults> results,
                  size_t dataSize, double error);


/**
 * Checks the correctness of the updates by recalculating all updated addresses and apply the same update again.
 * Since the update is a xor, the original values of the array can be recalculated with this method.
 * If a value differs from the original value it means there was an error during the benchmark execution.
 * 1% errors in the array are allowed.
 *
 * @param result_array The data array containing the data from the benchmark execution. It will be modified for the evaluation!
 * @param array_size The size of the data array
 */
double checkRandomAccessResults(HOST_DATA_TYPE* result_array, size_t array_size);


void generateInputData(HOST_DATA_TYPE* data, size_t dataSize);

#endif // SRC_HOST_RANDOM_ACCESS_FUNCTIONALITY_H_
