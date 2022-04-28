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

#include "linpack_data.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "communication_types.hpp"
#include "parameters.h"

linpack::LinpackProgramSettings::LinpackProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * (1 << (results["b"].as<uint>()))), blockSize(1 << (results["b"].as<uint>())), 
    isEmulationKernel(results.count("emulation") > 0), isDiagonallyDominant(results.count("uniform") == 0),
    torus_width(results["p"].as<uint>()) {
    int mpi_comm_rank;
    int mpi_comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
    // calculate the row and column of the MPI rank in the torus 
    if (mpi_comm_size % torus_width != 0) {
        throw std::runtime_error("MPI size not dividable by P=" + std::to_string(torus_width) + "!");
    } 
    torus_height = mpi_comm_size / torus_width;
    torus_row = (mpi_comm_rank / torus_width);
    torus_col = (mpi_comm_rank % torus_width);
}

std::map<std::string, std::string>
linpack::LinpackProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Emulate"] = (isEmulationKernel) ? "Yes" : "No";
        map["Data Type"] = STR(HOST_DATA_TYPE);
        map["FPGA Torus"] = "P=" + std::to_string(torus_width) + ", Q=" + std::to_string(torus_height);
        return map;
}

linpack::LinpackData::LinpackData(cl::Context context, size_t width, size_t height) : norma(0.0), context(context),
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

linpack::LinpackData::~LinpackData() {
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

/**
Standard LU factorization on a block with fixed size

Case 1 of Zhangs description
*/
void
linpack::gefa_ref(HOST_DATA_TYPE* a, unsigned n, unsigned lda, cl_int* ipvt) {
    for (int i = 0; i < n; i++) {
        ipvt[i] = i;
    }
    // For each diagnonal element
    for (int k = 0; k < n - 1; k++) {
        HOST_DATA_TYPE max_val = fabs(a[k * lda + k]);
        int pvt_index = k;
        for (int i = k + 1; i < n; i++) {
            if (max_val < fabs(a[k * lda + i])) {
                pvt_index = i;
                max_val = fabs(a[k * lda + i]);
            }
        }

        for (int i = k; i < n; i++) {
            HOST_DATA_TYPE tmp_val = a[i * lda + k];
            a[i * lda + k] = a[i * lda + pvt_index];
            a[i * lda + pvt_index] = tmp_val;
        }
        ipvt[k] = pvt_index;

        // For each element below it
        for (int i = k + 1; i < n; i++) {
            a[k * lda + i] *= -1.0 / a[k * lda + k];
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[j * lda + i] += a[k * lda + i] * a[j * lda + k];
            }
        }

#ifdef DEBUG
        std::cout << "A(k=" << k <<"): " << std::endl;
                for (int i= 0; i < n; i++) {
                    for (int j=0; j < n; j++) {
                        std::cout << a[i*lda + j] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout <<  std::endl;
#endif

    }
}

void
linpack::gesl_ref(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, cl_int* ipvt, unsigned n, unsigned lda) {
    auto b_tmp = new HOST_DATA_TYPE[n];
    {
        for (int k = 0; k < n; k++) {
            b_tmp[k] = b[k];
        }

        // solve l*y = b
        // For each row in matrix
        for (int k = 0; k < n - 1; k++) {
            if (ipvt[k] != k) {
                HOST_DATA_TYPE tmp = b_tmp[k];
                b_tmp[k] = b_tmp[ipvt[k]];
                b_tmp[ipvt[k]] = tmp;
            }
            // For each row below add
            for (int i = k + 1; i < n; i++) {
                // add solved upper row to current row
                b_tmp[i] += b_tmp[k] * a[lda * k + i];
            }
        }

        // now solve  u*x = y
        for (int k = n - 1; k >= 0; k--) {
            b_tmp[k] = b_tmp[k] / a[lda * k + k];
            for (int i = 0; i < k; i++) {
                b_tmp[i] -= b_tmp[k] * a[lda * k + i];
            }
        }
        for (int k = 0; k < n; k++) {
            b[k] = b_tmp[k];
        }
    }
    delete [] b_tmp;
}

void linpack::dmxpy(unsigned n1, HOST_DATA_TYPE* y, unsigned n2, unsigned ldm, HOST_DATA_TYPE* x, HOST_DATA_TYPE* m, bool transposed) {
    for (int i=0; i < n1; i++) {
        for (int j=0; j < n2; j++) {
            y[i] = y[i] + x[j] * (transposed ? m[ldm*i + j] :m[ldm*j + i]);
        }
    }
}

void
linpack::gefa_ref_nopvt(HOST_DATA_TYPE* a, unsigned n, unsigned lda) {
    // For each diagnonal element
    for (int k = 0; k < n; k++) {
        // Store negatie invers of diagonal elements to get rid of some divisions afterwards!
        a[k * lda + k] = -1.0 / a[k * lda + k];
        // For each element below it
        for (int i = k + 1; i < n; i++) {
            a[k * lda + i] *= a[k * lda + k];
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[j * lda + i] += a[k * lda + i] * a[j * lda + k];
            }
        }

#ifdef DEBUG
        std::cout << "A(k=" << k << "): " << std::endl;
                for (int i= 0; i < n; i++) {
                    for (int j=0; j < n; j++) {
                        std::cout << a[i*lda + j] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout <<  std::endl;
#endif

    }
}


void
linpack::gesl_ref_nopvt(HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, unsigned n, unsigned lda) {
    auto b_tmp = new HOST_DATA_TYPE[n];

    for (int k = 0; k < n; k++) {
        b_tmp[k] = b[k];
    }

    // solve l*y = b
    // For each row in matrix
    for (int k = 0; k < n - 1; k++) {
        // For each row below add
        for (int i = k + 1; i < n; i++) {
            // add solved upper row to current row
            b_tmp[i] += b_tmp[k] * a[lda * k + i];
        }
    }

    // now solve  u*x = y
    for (int k = n - 1; k >= 0; k--) {
        HOST_DATA_TYPE scale = b_tmp[k] * a[lda * k + k];
        b_tmp[k] = -scale;
        for (int i = 0; i < k; i++) {
            b_tmp[i] += scale * a[lda * k + i];
        }
    }
    for (int k = 0; k < n; k++) {
        b[k] = b_tmp[k];
    }
    delete [] b_tmp;
}
