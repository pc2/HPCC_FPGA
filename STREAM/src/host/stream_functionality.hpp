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

#ifndef SRC_HOST_NETWORK_FUNCTIONALITY_H_
#define SRC_HOST_NETWORK_FUNCTIONALITY_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "execution.h"
#include "cxxopts.hpp"
#include "setup/fpga_setup.hpp"
#include "parameters.h"


/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(std::shared_ptr<bm_execution::ExecutionTimings> results);


/**
 * Fill the data buffer with random number using the mersenne twister engine with
 * seed 0.
 *
 * @param data Data array that has to be filled
 * @param size Size of the data array that has to be filled
 */
void generateInputData(HOST_DATA_TYPE* A, HOST_DATA_TYPE* B, HOST_DATA_TYPE* C, unsigned array_size);


/**
 * Checks the calculation error of an FFt calculation by calculating the inverse FFT on the result data
 * and calculating the residual with abs(x - x')/(epsilon * log(FFT_SIZE)).
 *
 * @param verify_data The input data of the FFT calculation
 * @param result_data Result of the FFT calculation
 * @param iterations Number data iterations (total data size should be iterations * FFT_SIZE)
 * @return the residual error of the calculation
 */
double checkSTREAMResult(const HOST_DATA_TYPE* A, const HOST_DATA_TYPE* B, const HOST_DATA_TYPE* C, unsigned repetitions,
                         unsigned array_size);


#endif // SRC_HOST_NETWORK_FUNCTIONALITY_H_
