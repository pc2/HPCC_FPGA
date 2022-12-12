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
    matrixSize(results["b"].as<uint>() * results["m"].as<uint>()), blockSize(results["b"].as<uint>()),
    replicateInputBuffers(results["replicate-inputs"].count() > 0) {

}

std::map<std::string, std::string>
gemm::GEMMProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
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
            ("replicate-inputs", "Also replicates the input buffer for each kernel");
}

void
gemm::GEMMBenchmark::executeKernel(GEMMData &data) {
    timings = bm_execution::calculate(*executionSettings, data.A, data.B, data.C, data.C_out, data.alpha, data.beta);
}

void
gemm::GEMMBenchmark::collectResults() {

    uint number_measurements = timings.at("execution").size();
    std::vector<double> avg_measures(number_measurements);
#ifdef _USE_MPI_
    // Copy the object variable to a local variable to make it accessible to the lambda function
    int mpi_size = mpi_comm_size;
    MPI_Reduce(timings.at("execution").data(), avg_measures.data(), number_measurements, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::for_each(avg_measures.begin(),avg_measures.end(), [mpi_size](double& x) {x /= mpi_size;});
#else
    std::copy(timings.at("execution").begin(), timings.at("execution").end(), avg_measures.begin());
#endif
    if (mpi_comm_rank == 0) {
        // Calculate performance for kernel execution
        double tmean = 0;
        double tmin = std::numeric_limits<double>::max();

        double gflops = mpi_comm_size * 2.0 * (static_cast<double>(executionSettings->programSettings->matrixSize)
                            *static_cast<double>(executionSettings->programSettings->matrixSize)
                            *static_cast<double>(executionSettings->programSettings->matrixSize))/1.0e9;
        for (double currentTime : avg_measures) {
            tmean +=  currentTime;
            if (currentTime < tmin) {
                tmin = currentTime;
            }
        }
        tmean = tmean / avg_measures.size();
        results.emplace("t_mean", hpcc_base::HpccResult(tmean, "s"));
        results.emplace("t_min", hpcc_base::HpccResult(tmin, "s"));
        results.emplace("gflops", hpcc_base::HpccResult(gflops / tmin, "GFLOP/s"));
    }
}

void
gemm::GEMMBenchmark::printResults() {
    if (mpi_comm_rank == 0) {
        std::cout << std::left << std::setw(ENTRY_SPACE)
                << " best" << std::setw(ENTRY_SPACE) << " mean"
                << std::setw(ENTRY_SPACE) << " GFLOPS" << std::right << std::endl;

        std::cout << std::setw(ENTRY_SPACE)
                << results.at("t_min") << results.at("t_mean") << results.at("gflops")
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
gemm::GEMMBenchmark::validateOutput(gemm::GEMMData &data) {
    auto ref_data = generateInputData();

    gemm_ref(ref_data->A, ref_data->B, ref_data->C, executionSettings->programSettings->matrixSize, OPTIONAL_CAST(0.5), OPTIONAL_CAST(2.0));

    double resid = OPTIONAL_CAST(0.0);
    double normx = OPTIONAL_CAST(0.0);

    for (int i = 0; i < executionSettings->programSettings->matrixSize * executionSettings->programSettings->matrixSize; i++) {
        resid = (resid > fabs(data.C_out[i] - ref_data->C[i])) ? resid : fabs(data.C_out[i] - ref_data->C[i]);
        normx = (normx > fabs(data.C_out[i])) ? normx : fabs(data.C_out[i]);
    }

#ifdef _USE_MPI_
    double max_resid = 0.0;
    MPI_Reduce(&resid, &max_resid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    resid = max_resid;
#endif

    // Calculate the overall error only on rank 0
    if (mpi_comm_rank == 0) {
        // Calculate the residual error normalized to the total matrix size, input values and machine epsilon
        double eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
        double residn = resid / (executionSettings->programSettings->matrixSize*executionSettings->programSettings->matrixSize*ref_data->normtotal*normx*eps);

        errors.emplace("epsilon", hpcc_base::HpccResult(eps, ""));
        errors.emplace("residual", hpcc_base::HpccResult(resid, ""));
        errors.emplace("residual_norm", hpcc_base::HpccResult(residn, ""));

        return residn < 1.0;
    }
    // All other ranks are always reporting success of the validation
    return true;
}

void
gemm::GEMMBenchmark::printError() {
    if (mpi_comm_rank == 0) {
        std::cout << std::left << std::setw(ENTRY_SPACE) << " norm. residual" << std::setw(ENTRY_SPACE) << " res. error" << std::setw(ENTRY_SPACE) << " mach. eps" << std::right << std::endl;
        std::cout << errors.at("residual_norm") << errors.at("residual") << errors.at("epsilon") << std::endl;
    }
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
        #pragma omp parallel
        {
        #pragma omp for
        for (int i=0; i < n; i++) {
            for (int j=0; j < n; j++) {
                c[i * n + j] = beta * c[i*n + j];
            }
        }

#define HOST_MM_BLOCK_SIZE 256

        #pragma omp for collapse(2)
        for (int i=0; i < n; i+=HOST_MM_BLOCK_SIZE) {
            for (int j=0; j < n; j+=HOST_MM_BLOCK_SIZE) {
                for (int k=0; k < n; k+=HOST_MM_BLOCK_SIZE) {
                     for (int ii=i; ii < std::min(i + HOST_MM_BLOCK_SIZE, n); ii++) {
                        for (int kk=k; kk < std::min(k + HOST_MM_BLOCK_SIZE, n); kk++) {  
                            HOST_DATA_TYPE scaled_a =  alpha * a[ii*n + kk];
                            for (int jj=j; jj < std::min(j + HOST_MM_BLOCK_SIZE, n); jj++) {   
                                c[ii*n + jj] += scaled_a * b[kk*n + jj];
                            }
                        }
                     }
                }
            }
        }
        }
#endif
}
