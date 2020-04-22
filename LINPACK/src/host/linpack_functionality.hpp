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
#include "linpack_functionality.hpp"
#include "execution.h"
#include "parameters.h"

/*
Short description of the program.
Moreover the version and build time is also compiled into the description.
*/
#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define PROGRAM_DESCRIPTION "Implementation of the LINPACK benchmark"\
                            " proposed in the HPCC benchmark suite for FPGA."\

#define ENTRY_SPACE 15

struct ProgramSettings {
    uint numRepetitions;
    uint matrixSize;
    int defaultPlatform;
    int defaultDevice;
    std::string kernelFileName;
};


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
parseProgramParameters(int argc, char *argv[]);

/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(std::shared_ptr<bm_execution::ExecutionTimings> results, unsigned matrix_size);

/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings retrieved from the command line
 * @param device The device used for execution
 */
void printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                             const cl::Device &device);


/**
 * Fill the data buffer with random number using the mersenne twister engine with
 * seed 0.
 *
 * @param data Data array that has to be filled
 * @param size Size of the data array that has to be filled
 */
void generateInputData(HOST_DATA_TYPE* A, HOST_DATA_TYPE* b, cl_int* ipvt, unsigned matrix_size, HOST_DATA_TYPE* norma);



double checkLINPACKresults(const HOST_DATA_TYPE* b_res, unsigned n);

/**
 *
 *
 * @param n1
 * @param y
 * @param n2
 * @param ldm
 * @param x
 * @param m
 */
void dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m);

/**
Gaussian elemination reference implementation without pivoting.
Can be used in exchange with kernel functions for functionality testing

@param a the matrix with size of n*n
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gefa_ref(HOST_DATA_TYPE* a, unsigned n, unsigned lda, cl_int* ipvt);

/**
Solve linear equations using its LU decomposition.
Therefore solves A*x = b by solving L*y = b and then U*x = y with A = LU
where A is a matrix of size n*n

@param a the matrix a in LU representation calculated by gefa call
@param b vector b of the given equation
@param ipvt vector containing pivoting information
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gesl_ref(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, cl_int* ipvt, unsigned n, unsigned lda);


#endif // SRC_HOST_NETWORK_FUNCTIONALITY_H_
