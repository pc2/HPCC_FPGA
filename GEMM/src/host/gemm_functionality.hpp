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
#ifndef COMMON_FUNCTIONALITY_H
#define COMMON_FUNCTIONALITY_H

/* C++ standard library headers */
#include <memory>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

/*
Number of times the execution of the benchmark will be repeated.
*/
#ifndef NTIMES
#define NTIMES 1
#endif

/*
Prefix of the function name of the used kernel.
It will be used to construct the full function name for the case of replications.
The full name will be
*/
#define GEMM_KERNEL "gemm"


#ifdef _USE_BLAS_

extern "C" void sgemm_(char*, char*, int*, int*,int*, float*, float*, int*, float*, int*, float*, float*, int*);

#endif

double
checkGEMMresults(HOST_DATA_TYPE* c_res, cl_int lda, cl_int n);

/**
Print the benchmark results to stdout

@param results the struct containing the results of the benchmark execution
@param matrixSize size of the calculated matrix
*/
void printResults(std::shared_ptr<bm_execution::ExecutionTimings> results,
                  size_t matrixSize);

/**
Generate a matrix using pseudo random numbers with fixed seed.
Generates inuts for th


@param a pointer to the matrix
@param seed Seed for the pseudo random number generation
@param lda width of a row in the matrix
@param n number of rows in the matrix
@param norma the maximum value in the matrix A that can be used to calculate the residual error
*/
void matgen(HOST_DATA_TYPE* a, int seed, cl_int lda, cl_int n, HOST_DATA_TYPE* norma);

/**
Multiply matrix with a vector and add it to another vector.

C = alpha * A * B + beta * C

@param a matrix A
@param b matrix B
@param c matrix C that will also be the result matrix
@param n size of all quadratic matrices
@param alpha scalar value used to scale A * B
@param beta scalar value used to scale C
*/
void gemm_ref( HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
                                int n, HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta);

HOST_DATA_TYPE epslon (HOST_DATA_TYPE x);

#endif // SRC_HOST_COMMON_FUNCTIONALITY_H_
