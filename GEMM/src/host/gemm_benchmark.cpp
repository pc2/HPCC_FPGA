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

#include "gemm_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "parameters.h"

gemm::GEMMProgramSettings::GEMMProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["b"].as<uint>() * results["m"].as<uint>()), blockSize(results["b"].as<uint>()), kernelReplications(results["r"].as<uint>()),
    replicateInputBuffers(results["replicate-inputs"].count() > 0) {

}

std::map<std::string, std::string>
gemm::GEMMProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Kernel Replications"] = std::to_string(kernelReplications);
        map["Replicate Inputs"] = replicateInputBuffers ? "Yes" : "No";
        return map;
}

gemm::GEMMData::GEMMData(cl::Context context, uint size) : normtotal(0.0), alpha(0.5), beta(2.0), context(context) {
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    B = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    C = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
    C_out = reinterpret_cast<HOST_DATA_TYPE*>(
                        clSVMAlloc(context(), 0 ,
                        size * size * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&A), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C_out), 4096, size * size * sizeof(HOST_DATA_TYPE));
#endif
}

gemm::GEMMData::~GEMMData() {
#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void**>(A));
    clSVMFree(context(), reinterpret_cast<void**>(B));
    clSVMFree(context(), reinterpret_cast<void**>(C));
    clSVMFree(context(), reinterpret_cast<void**>(C_out));
#else
    free(A);
    free(B);
    free(C);
    free(C_out);
#endif
}

gemm::GEMMBenchmark::GEMMBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
}

void
gemm::GEMMBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
            ("m", "Matrix size in number of blocks in a single dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
            ("b", "Block size in number of values in one dimension",
             cxxopts::value<cl_uint>()->default_value(std::to_string(BLOCK_SIZE)))
            ("r", "Number of used kernel replications",
             cxxopts::value<cl_uint>()->default_value(std::to_string(NUM_REPLICATIONS)))
            ("replicate-inputs", "Also replicates the input buffer for each kernel");
}

std::unique_ptr<gemm::GEMMExecutionTimings>
gemm::GEMMBenchmark::executeKernel(GEMMData &data) {
    return bm_execution::calculate(*executionSettings, data.A, data.B, data.C, data.C_out, data.alpha, data.beta);
}

void
gemm::GEMMBenchmark::collectAndPrintResults(const gemm::GEMMExecutionTimings &output) {

    uint number_measurements = output.timings.size();
    std::vector<double> avg_measures(number_measurements);
#ifdef _USE_MPI_
    MPI_Reduce(output.timings.data(), avg_measures.data(), number_measurements, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::accumulate(avg_measures.begin(), avg_measures.end(),number_measurements, std::divides<double>());
#else
    std::copy(output.timings.begin(), output.timings.end(), avg_measures.begin());
#endif
    if (mpi_comm_rank == 0) {
        std::cout << std::setw(ENTRY_SPACE)
                << "best" << std::setw(ENTRY_SPACE) << "mean"
                << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

        // Calculate performance for kernel execution plus data transfer
        double tmean = 0;
        double tmin = std::numeric_limits<double>::max();

        double gflops = 2.0 * (static_cast<double>(executionSettings->programSettings->matrixSize)
                            *static_cast<double>(executionSettings->programSettings->matrixSize)
                            *static_cast<double>(executionSettings->programSettings->matrixSize))/1.0e9;
        for (double currentTime : avg_measures) {
            tmean +=  currentTime;
            if (currentTime < tmin) {
                tmin = currentTime;
            }
        }
        tmean = tmean / avg_measures.size();

        std::cout << std::setw(ENTRY_SPACE)
                << tmin << std::setw(ENTRY_SPACE) << tmean
                << std::setw(ENTRY_SPACE) << gflops / tmin
                << std::endl;
    }
}

std::unique_ptr<gemm::GEMMData>
gemm::GEMMBenchmark::generateInputData() {
    auto d = std::unique_ptr<gemm::GEMMData>(new gemm::GEMMData(*executionSettings->context, executionSettings->programSettings->matrixSize));
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            d->A[executionSettings->programSettings->matrixSize*i+j] = OPTIONAL_CAST(static_cast<double>(dis(gen)));
            d->B[executionSettings->programSettings->matrixSize*i+j] = OPTIONAL_CAST(static_cast<double>(dis(gen)));
            d->C[executionSettings->programSettings->matrixSize*i+j] = OPTIONAL_CAST(static_cast<double>(dis(gen)));
            d->C_out[executionSettings->programSettings->matrixSize*i+j] = OPTIONAL_CAST(0.0);
            d->normtotal = std::max(std::max(d->normtotal, d->A[executionSettings->programSettings->matrixSize*i+j]), 
                                        std::max(d->B[executionSettings->programSettings->matrixSize*i+j], 
                                        d->C[executionSettings->programSettings->matrixSize*i+j]));
        }
    }
    return d;
}

bool  
gemm::GEMMBenchmark::validateOutputAndPrintError(gemm::GEMMData &data) {
    auto ref_data = generateInputData();

    gemm_ref(ref_data->A, ref_data->B, ref_data->C, executionSettings->programSettings->matrixSize, OPTIONAL_CAST(0.5), OPTIONAL_CAST(2.0));

    double resid = OPTIONAL_CAST(0.0);
    double normx = OPTIONAL_CAST(0.0);

    for (int i = 0; i < executionSettings->programSettings->matrixSize * executionSettings->programSettings->matrixSize; i++) {
        resid = (resid > fabs(data.C_out[i] - ref_data->C[i])) ? resid : fabs(data.C_out[i] - ref_data->C[i]);
        normx = (normx > fabs(data.C_out[i])) ? normx : fabs(data.C_out[i]);
    }

    double eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    double residn = resid / (executionSettings->programSettings->matrixSize*executionSettings->programSettings->matrixSize*ref_data->normtotal*normx*eps);

#ifdef _USE_MPI_
    double max_norm_resid = 0.0;
    MPI_Reduce(&residn, &max_norm_resid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    residn = max_norm_resid;
#endif

    if (mpi_comm_rank == 0) {
        std::cout << "  norm. resid        resid       "\
                    "machep" << std::endl;
        std::cout << std::setw(ENTRY_SPACE) << residn << std::setw(ENTRY_SPACE)
                << resid << std::setw(ENTRY_SPACE) << eps
                << std::endl;
    }
    return residn < 1.0;
}

void 
gemm::gemm_ref(HOST_DATA_TYPE* a,HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
                                int n, HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta) {
#ifdef _USE_BLAS_
    char ta = 'N';
    char tb = 'N';
#endif
#if (defined(_USE_BLAS_) && DATA_TYPE_SIZE == 2) 
        // convert matrices to single precision to allow the use of BLAS routine
        std::unique_ptr<float[]> temp_a = std::unique_ptr<float[]>(new float[n * n]);
        std::unique_ptr<float[]> temp_b = std::unique_ptr<float[]>(new float[n * n]);
        std::unique_ptr<float[]> temp_c = std::unique_ptr<float[]>(new float[n * n]);
        for (int i=0; i < n; i++) {
            for (int j=0; j < n; j++) {
                temp_a[i * n + j] = half_float::half_cast<float, half_float::half>(a[i*n + j]);
                temp_b[i * n + j] = half_float::half_cast<float, half_float::half>(b[i*n + j]);
                temp_c[i * n + j] = half_float::half_cast<float, half_float::half>(c[i*n + j]);
            }
        }
        float alpha_sp = half_float::half_cast<float, half_float::half>(alpha);
        float beta_sp = half_float::half_cast<float, half_float::half>(beta);
        // Use single precision for validation
        sgemm_(&ta, &tb, &n, &n, &n, &alpha_sp, temp_b.get(), &n, temp_a.get(), &n, &beta_sp, temp_c.get(), &n);
        // convert the result back to half precision
        for (int i=0; i < n; i++) {
            for (int j=0; j < n; j++) {
                c[i * n + j] = half_float::half_cast<half_float::half, float>(temp_c[i*n + j]);
            }
        }
#endif
#if (defined(_USE_BLAS_) && DATA_TYPE_SIZE == 4) 
        // Use single precision for validation
        sgemm_(&ta, &tb, &n, &n, &n, &alpha, b, &n, a, &n, &beta, c, &n);
#endif
#if (defined(_USE_BLAS_) && DATA_TYPE_SIZE == 8) 
        // use double precision for validation
        dgemm_(&ta, &tb, &n, &n, &n, &alpha, b, &n, a, &n, &beta, c, &n);
#endif
#if (!defined(_USE_BLAS_) || (DATA_TYPE_SIZE != 2 && DATA_TYPE_SIZE != 4 && DATA_TYPE_SIZE != 8)) 
        // Caclulate manually. Thisi s the default, if BLAS is not found
        for (int i=0; i < n; i++) {
            for (int j=0; j < n; j++) {
                c[i * n + j] = beta * c[i*n + j];
            }
        }

        for (int i=0; i < n; i++) {
            for (int j=0; j < n; j++) {
                for (int k=0; k < n; k++) {
                    c[i*n + j] += alpha * a[i*n + k] * b[k*n + j];
                }
            }
        }
#endif
}
