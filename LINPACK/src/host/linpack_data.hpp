/*
Copyright (c) 2022 Marius Meyer

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

#ifndef SRC_HOST_LINPACK_DATA_H_
#define SRC_HOST_LINPACK_DATA_H_

/* C++ standard library headers */
#include <complex>
#include <memory>
#include <random>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"
extern "C" {
    #include "gmres.h"
}

/**
 * @brief Contains all classes and methods needed by the LINPACK benchmark
 * 
 */
namespace linpack {

/**
 * @brief The Linpack specific program settings
 * 
 */
class LinpackProgramSettings : public hpcc_base::BaseSettings {

public:
    /**
     * @brief The size of the local matrix in number of blocks in one dimension
     * 
     */
    uint matrixSize;

    /**
     * @brief Size of a single block of the matrix in values in one dimension
     * 
     */
    uint blockSize;

    /**
     * @brief Indicates if the generated input matrix should be diagonally dominant
     * 
     */
    bool isDiagonallyDominant;

    /**
     * @brief True, if the used kernel is an emulation kernel. Different kernel arguments may be used in this case to
     *          simulate persistent local memory.
     * 
     */
    bool isEmulationKernel;

    /**
     * @brief The row position of this MPI rank in the torus
     * 
     */
    int torus_row;

    /**
     * @brief The rcolumn position of this MPI rank in the torus
     * 
     */
    int torus_col;

    /**
     * @brief Width of the torus in number of ranks
     * 
     */
    int torus_width;

    /**
     * @brief Height of the FPGA torus in number of ranks
     * 
     */
    int torus_height;

    /**
     * @brief Construct a new Linpack Program Settings object
     * 
     * @param results the result map from parsing the program input parameters
     */
    LinpackProgramSettings(cxxopts::ParseResult &results);

    /**
     * @brief Get a map of the settings. This map will be used to print the final configuration.
     * 
     * @return a map of program parameters. keys are the name of the parameter.
     */
    std::map<std::string, std::string> getSettingsMap() override;

};

/**
 * @brief Data class containing the data the kernel is exeucted with
 * 
 */
template<class TContext>
class LinpackData {

public:

    /**
     * @brief  The input matrix representing the left side of the linear equation system
     * 
     */
    HOST_DATA_TYPE *A;

    /**
     * @brief  The input vector the right side of the linear equation system
     * 
     */
    HOST_DATA_TYPE *b;

    /**
     * @brief A vector that can be used to store pivoting information
     * 
     */
    cl_int* ipvt;

    /**
     * @brief Width of the local matrix in values
     * 
     */
    size_t matrix_width;

    /**
     * @brief Height of the local matrix in values
     * 
     */
    size_t matrix_height;

    /**
     * @brief The context that is used to allocate memory in SVM mode
     * 
     */
    TContext context;

    /**
     * @brief The maximum value of A that will be used for the error calculation
     * 
     */
    HOST_DATA_TYPE norma;

    /**
     * @brief The maximum value of A that will be used for the error calculation
     * 
     */
    HOST_DATA_TYPE normb;

    /**
     * @brief Construct a new Linpack Data object
     * 
     * @param context The OpenCL context used to allocate memory in SVM mode
     * @param width width of the local matrix in values
     * @param height height of the local matrix in values
     */
    LinpackData(TContext &context, size_t width, size_t height) : norma(0.0), 
#ifdef USE_SVM
    context(context),
#endif
    matrix_width(width), matrix_height(height) {
#ifdef USE_SVM
        A = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            size * size * sizeof(HOST_DATA_TYPE), 1024));
        b = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            size  * sizeof(HOST_DATA_TYPE), 1024));
        ipvt = reinterpret_cast<cl_int*>(
                            clSVMAlloc(context(), 0 ,
                            size * sizeof(cl_int), 1024));
#else
        posix_memalign(reinterpret_cast<void**>(&A), 4096, width * height * sizeof(HOST_DATA_TYPE));
        posix_memalign(reinterpret_cast<void**>(&b), 4096, width * sizeof(HOST_DATA_TYPE));
        posix_memalign(reinterpret_cast<void**>(&ipvt), 4096, height * sizeof(cl_int));
#endif
    }

    ~LinpackData() {
#ifdef USE_SVM
        clSVMFree(context(), reinterpret_cast<void*>(A));
        clSVMFree(context(), reinterpret_cast<void*>(b));
        clSVMFree(context(), reinterpret_cast<void*>(ipvt));
#else
        free(A);
        free(b);
        free(ipvt);
#endif
    }

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class LinpackExecutionTimings {
public:
    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution for the gefa kernel
     * 
     */
    std::vector<double> gefaTimings;

    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution for the gesl kernel
     * 
     */
    std::vector<double> geslTimings;


};

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
void dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m, bool transposed);

/**
Gaussian elemination reference implementation with partial pivoting.
Can be used in exchange with kernel functions for functionality testing

@param a the matrix with size of n*n
@param n size of matrix A
@param lda row with of the matrix. must be >=n
@param ipvt array of pivoting indices

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

/**
Gaussian elemination reference implementation without pivoting.
Can be used in exchange with kernel functions for functionality testing

@param a the matrix with size of n*n
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gefa_ref_nopvt(HOST_DATA_TYPE* a, unsigned n, unsigned lda);

/**
Solve linear equations using its LU decomposition without pivoting.
Therefore solves A*x = b by solving L*y = b and then U*x = y with A = LU
where A is a matrix of size n*n

@param a the matrix a in LU representation calculated by gefa call
@param b vector b of the given equation
@param n size of matrix A
@param lda row with of the matrix. must be >=n

*/
void gesl_ref_nopvt(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, unsigned n, unsigned lda);


}
#endif // SRC_HOST_LINPACK_DATA_H__
