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

#include "linpack_benchmark.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "communication_types.hpp"
#include "execution_types/execution_types.hpp"
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

linpack::LinpackBenchmark::LinpackBenchmark(int argc, char* argv[]) : HpccFpgaBenchmark(argc, argv) {
    setupBenchmark(argc, argv);
}

void
linpack::LinpackBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("m", "Global matrix size in number of blocks in one dimension. Local matrix sizes will be determined by PQ grid.",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("b", "Log2 of the block size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(LOCAL_MEM_BLOCK_LOG)))
        ("p", "Width of the FPGA grid. The heigth (Q) will be calculated from mpi_size / P.",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_P_VALUE)))
        ("uniform", "Generate a uniform matrix instead of a diagonally dominant. This has to be supported by the FPGA kernel!")
        ("emulation", "Use kernel arguments for emulation. This may be necessary to simulate persistent local memory on the FPGA");
}

void
linpack::LinpackBenchmark::executeKernel(LinpackData &data) {
    switch (executionSettings->programSettings->communicationType) {
        case hpcc_base::CommunicationType::pcie_mpi : timings = execution::pcie::calculate(*executionSettings, data); break;
        case hpcc_base::CommunicationType::intel_external_channels: timings = execution::iec::calculate(*executionSettings, data); break;
        default: throw std::runtime_error("No calculate method implemented for communication type " + commToString(executionSettings->programSettings->communicationType));
    }
#ifdef DISTRIBUTED_VALIDATION
    distributed_gesl_nopvt_ref(data);
#endif
}

void
linpack::LinpackBenchmark::collectResults() {
    // Calculate performance for kernel execution plus data transfer
    double t = 0;
    double tlu = 0;
    double tsl = 0;
    double tmin = std::numeric_limits<double>::max();
    double lu_min = std::numeric_limits<double>::max();
    double sl_min = std::numeric_limits<double>::max();

#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": Result collection started" << std::endl;
#endif

    std::vector<double> global_lu_times(timings["gefa"].size());
    MPI_Reduce(timings["gefa"].data(), global_lu_times.data(), timings["gefa"].size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> global_sl_times(timings["gesl"].size());
    MPI_Reduce(timings["gesl"].data(), global_sl_times.data(), timings["gesl"].size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": Result collection done" << std::endl;
#endif


    if (mpi_comm_rank > 0) {
        // Only the master rank needs to calculate and print result
        return;
    }

    double total_matrix_size = static_cast<double>(executionSettings->programSettings->matrixSize);
    double gflop_lu = ((2.0e0*total_matrix_size * total_matrix_size * total_matrix_size)/ 3.0) / 1.0e9; 
    double gflop_sl = (2.0*(total_matrix_size * total_matrix_size))/1.0e9;
    for (int i =0; i < global_lu_times.size(); i++) {
        double currentTime = global_lu_times[i] + global_sl_times[i];
        t +=  currentTime;
        tlu +=  global_lu_times[i];
        tsl += global_sl_times[i];
        if (currentTime < tmin) {
            tmin = currentTime;
        }
        if (global_lu_times[i] < lu_min) {
            lu_min = global_lu_times[i];
        }
        if (global_sl_times[i] < sl_min) {
            sl_min = global_sl_times[i];
        }
    }
    
    results.emplace("t_mean", hpcc_base::HpccResult(t / global_lu_times.size(), "s"));
    results.emplace("t_min", hpcc_base::HpccResult(tmin, "s"));
    results.emplace("tlu_mean", hpcc_base::HpccResult(tlu / global_lu_times.size(), "s"));
    results.emplace("tlu_min", hpcc_base::HpccResult(lu_min, "s"));
    results.emplace("tsl_mean", hpcc_base::HpccResult(tsl / global_sl_times.size(), "s"));
    results.emplace("tsl_min", hpcc_base::HpccResult(sl_min, "s"));
    results.emplace("gflops", hpcc_base::HpccResult((gflop_lu + gflop_sl) / tmin, "GFLOP/s"));
    results.emplace("gflops_lu", hpcc_base::HpccResult(gflop_lu / lu_min, "GFLOP/s"));
    results.emplace("gflops_sl", hpcc_base::HpccResult(gflop_sl / sl_min, "GFLOP/s"));
    
    return;
}

void
linpack::LinpackBenchmark::printResults() {
    if (mpi_comm_rank == 0) {
        std::cout << std::left << std::setw(ENTRY_SPACE) << " Method"
            << std::setw(ENTRY_SPACE) << " best"
            << std::setw(ENTRY_SPACE) << " mean"
            << std::setw(ENTRY_SPACE) << " GFLOPS"
            << std::endl;

        std::cout << std::left << std::setw(ENTRY_SPACE) << " total" 
                  << results.at("t_min") << results.at("t_mean") << results.at("gflops")
                  << std::endl;

        std::cout << std::left << std::setw(ENTRY_SPACE) << " GEFA"
                << results.at("tlu_min") << results.at("tlu_mean") << results.at("gflops_lu")
                << std::endl;

        std::cout << std::left << std::setw(ENTRY_SPACE) << " GESL"
                  << results.at("tsl_min") << results.at("tsl_mean") << results.at("gflops_sl")
                  << std::right << std::endl;
    }
}

std::unique_ptr<linpack::LinpackData>
linpack::LinpackBenchmark::generateInputData() {
    int local_matrix_width = executionSettings->programSettings->matrixSize / executionSettings->programSettings->torus_width;
    int local_matrix_height = executionSettings->programSettings->matrixSize / executionSettings->programSettings->torus_height;

    if ((executionSettings->programSettings->matrixSize / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width > 0 || 
        (executionSettings->programSettings->matrixSize / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_height > 0) {
            throw std::runtime_error("Global matrix size must be multiple of LCM of PQ grid!");
    }

    auto d = std::unique_ptr<linpack::LinpackData>(new linpack::LinpackData(*executionSettings->context ,local_matrix_width, local_matrix_height));
    std::mt19937 gen(this->mpi_comm_rank);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    d->norma = 0.0;
    d->normb = 0.0;


    /*
    Generate a matrix by using pseudo random number in the range (0,1)
    */
    for (int j = 0; j < local_matrix_height; j++) {
        // fill a single column of the matrix
        for (int i = 0; i < local_matrix_width; i++) {
                HOST_DATA_TYPE temp = dis(gen);
                d->A[local_matrix_width*j+i] = temp;
                d->norma = (temp > d->norma) ? temp : d->norma;
        }
    }


    // If the matrix should be diagonally dominant, we need to exchange the sum of the rows with
    // the ranks that share blocks in the same column
    if (executionSettings->programSettings->isDiagonallyDominant) {
        // create a communicator to exchange the rows
        MPI_Comm row_communicator;
        MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_row, 0,&row_communicator);

        // Caclulate the sum for every row and insert in into the matrix
        for (int local_matrix_row = 0; local_matrix_row < local_matrix_height; local_matrix_row++) {
            int blockSize = executionSettings->programSettings->blockSize;
            int global_matrix_row = executionSettings->programSettings->torus_row * blockSize + (local_matrix_row / blockSize) * blockSize * executionSettings->programSettings->torus_height + (local_matrix_row % blockSize);
            int local_matrix_col = (global_matrix_row - executionSettings->programSettings->torus_col * blockSize) / (blockSize * executionSettings->programSettings->torus_width) * blockSize + (global_matrix_row % blockSize);
            int diagonal_rank = (global_matrix_row / blockSize) % executionSettings->programSettings->torus_width;
            bool diagonal_on_this_rank = diagonal_rank == executionSettings->programSettings->torus_col;
            // set the diagonal elements of the matrix to 0
            if (diagonal_on_this_rank) {
                d->A[local_matrix_width*local_matrix_row + local_matrix_col] = 0.0;
            }
            HOST_DATA_TYPE local_row_sum = 0.0;
            for (int i = 0; i < local_matrix_width; i++) {
                local_row_sum += d->A[local_matrix_width*local_matrix_row + i];
            } 
            HOST_DATA_TYPE row_sum = 0.0;
            MPI_Reduce(&local_row_sum, &row_sum, 1, MPI_DATA_TYPE, MPI_SUM, diagonal_rank, row_communicator);
            // insert row sum into matrix if it contains the diagonal block
            if (diagonal_on_this_rank) {
                // update norm of local matrix
                d->norma = (row_sum > d->norma) ? row_sum : d->norma;
                d->A[local_matrix_width*local_matrix_row + local_matrix_col] = row_sum;
            }
        }
    }
        
    // initialize other vectors
    for (int i = 0; i < local_matrix_width; i++) {
        d->b[i] = 0.0;
    }
    for (int i = 0; i < local_matrix_height; i++) {
        d->ipvt[i] = i;
    }

    MPI_Comm col_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_col, 0,&col_communicator);

    // Generate vector b by accumulating the columns of the matrix.
    // This will lead to a result vector x with ones on every position
    // Every rank will have a valid part of the final b vector stored
    for (int j = 0; j < local_matrix_width; j++) {
        HOST_DATA_TYPE local_col_sum = 0.0;
        for (int i = 0; i < local_matrix_height; i++) {
            local_col_sum += d->A[local_matrix_width*i+j];
        }
        MPI_Allreduce(&local_col_sum, &(d->b[j]), 1, MPI_DATA_TYPE, MPI_SUM, col_communicator);   
        d->normb = (d->b[j] > d->normb) ? d->b[j] : d->normb;   
    }
    return d;
}

bool  
linpack::LinpackBenchmark::validateOutput(linpack::LinpackData &data) {
    uint n= executionSettings->programSettings->matrixSize;
    uint matrix_width = data.matrix_width;
    uint matrix_height = data.matrix_height;
    double residn;
    double resid = 0.0;
    double normx = 0.0;
#ifndef DISTRIBUTED_VALIDATION
    if (mpi_comm_rank > 0) {
        for (int j = 0; j < matrix_height; j++) {
            for (int i = 0; i < matrix_width; i+= executionSettings->programSettings->blockSize) {
                MPI_Send(&data.A[matrix_width * j + i], executionSettings->programSettings->blockSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
            }
        }
        if (executionSettings->programSettings->torus_row == 0) {
            for (int i = 0; i < matrix_width; i+= executionSettings->programSettings->blockSize) {
                MPI_Send(&data.b[i], executionSettings->programSettings->blockSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
            }
        }
        residn = 0;
    }
    else {
        MPI_Status status;
        size_t current_offset = 0;
        std::vector<HOST_DATA_TYPE> total_b_original(n);
        std::vector<HOST_DATA_TYPE> total_b(n);
        std::vector<HOST_DATA_TYPE> total_a(n*n);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i+= executionSettings->programSettings->blockSize) {
                int recvcol= (i / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width;
                int recvrow= (j / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_height;
                int recvrank = executionSettings->programSettings->torus_width * recvrow + recvcol;
                if (recvrank > 0) {
                    MPI_Recv(&total_a[j * n + i],executionSettings->programSettings->blockSize, MPI_DATA_TYPE, recvrank, 0, MPI_COMM_WORLD,  &status);
                }
                else {
                    for (int k=0; k < executionSettings->programSettings->blockSize; k++) {
                        total_a[j * n + i + k] = data.A[current_offset + k];
                    }
                    current_offset += executionSettings->programSettings->blockSize;
                }
            }
        }
        current_offset = 0;
        for (int i = 0; i < n; i+= executionSettings->programSettings->blockSize) {
            int recvcol= (i / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width;
            if (recvcol > 0) {
                MPI_Recv(&total_b[i], executionSettings->programSettings->blockSize, MPI_DATA_TYPE, recvcol, 0, MPI_COMM_WORLD, &status);
            }
            else {
                for (int k=0; k < executionSettings->programSettings->blockSize; k++) {
                    total_b[i + k] = data.b[current_offset + k];
                }
                current_offset += executionSettings->programSettings->blockSize;
            }
        }

        std::copy(total_b.begin(), total_b.end(), total_b_original.begin());
        gesl_ref_nopvt(total_a.data(), total_b.data(), n, n);

        for (int i = 0; i < n; i++) {
            resid = (resid > std::abs(total_b[i] - 1)) ? resid : std::abs(total_b[i] - 1);
            normx = (normx > std::abs(total_b_original[i])) ? normx : std::abs(total_b_original[i]);
        }
    }
#else
    double local_resid = 0;
    double local_normx = data.normb;
    #pragma omp parallel for reduction(max:local_resid)
    for (int i = 0; i < data.matrix_width; i++) {
        local_resid = (local_resid > std::abs(data.b[i] - 1)) ? local_resid : std::abs(data.b[i] - 1);
    }
#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": resid=" << local_resid << ", normx=" << local_normx << std::endl;
#endif

    MPI_Reduce(&local_resid, &resid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_normx, &normx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#endif


    HOST_DATA_TYPE eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    residn = resid / (static_cast<double>(n)*normx*eps);

    #ifndef NDEBUG
        if (residn > 1 &&  mpi_comm_size == 1) {
            auto ref_result = generateInputData();
            // For each column right of current diagonal element
            for (int j = 0; j < n; j++) {
                // For each element below it
                for (int i = 0; i < n; i++) {
                    std::cout << ref_result->A[n * j + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            // For each column right of current diagonal element
            for (int j = 0; j < n; j++) {
                // For each element below it
                for (int i = 0; i < n; i++) {
                    std::cout << data.A[n * j + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            if (executionSettings->programSettings->isDiagonallyDominant) {
                linpack::gefa_ref_nopvt(ref_result->A, n, n);
                linpack::gesl_ref_nopvt(ref_result->A, ref_result->b, n, n);
            }
            else {
                linpack::gefa_ref(ref_result->A, n, n, ref_result->ipvt);
                linpack::gesl_ref(ref_result->A, ref_result->b, ref_result->ipvt, n, n);
            }
            // For each column right of current diagonal element
            for (int j = 0; j < n; j++) {
                // For each element below it
                for (int i = 0; i < n; i++) {
                    std::cout << std::abs(ref_result->A[n * j + i] - data.A[n * j + i]) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    #endif

    errors.emplace("epsilon", hpcc_base::HpccResult(eps, ""));
    errors.emplace("residual", hpcc_base::HpccResult(resid, ""));
    errors.emplace("residual_norm", hpcc_base::HpccResult(residn, ""));

    if (mpi_comm_rank == 0) {
        return residn < 1;
    } else {
        return true;
    }
}

void
linpack::LinpackBenchmark::printError() {
    if (mpi_comm_rank == 0) {
        std::cout << std::left << std::setw(ENTRY_SPACE) << " norm. residual" << std::setw(ENTRY_SPACE) << " res. error" << std::setw(ENTRY_SPACE) << " mach. eps" << std::right << std::endl;
        std::cout << errors.at("residual_norm") << errors.at("residual") << errors.at("epsilon") << std::endl;
    }
}

void 
linpack::LinpackBenchmark::distributed_gesl_nopvt_ref(linpack::LinpackData& data) {
    uint global_matrix_size = executionSettings->programSettings->matrixSize;
    uint matrix_width = data.matrix_width;
    uint matrix_height = data.matrix_height;
    uint block_size = executionSettings->programSettings->blockSize;
    // create a communicator to exchange the rows
    MPI_Comm row_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_row, 0,&row_communicator);
    MPI_Comm col_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_col, 0,&col_communicator);
    std::vector<HOST_DATA_TYPE> b_tmp(matrix_width);

    for (int k = 0; k < b_tmp.size(); k++) {
        b_tmp[k] = data.b[k];
    }

    // solve l*y = b
    // For each row in matrix
    for (int k = 0; k < global_matrix_size - 1; k++) {
        size_t local_k_index_col =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * executionSettings->programSettings->torus_height) * block_size;
        size_t remaining_k_col = k % (block_size * executionSettings->programSettings->torus_width);
        size_t remaining_k_row = k % (block_size * executionSettings->programSettings->torus_height);
        size_t start_offset = local_k_index_col;
        if (remaining_k_col / block_size > executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
            start_offset = local_k_index_col;
        }
        else if (remaining_k_col / block_size == executionSettings->programSettings->torus_col) {
            local_k_index_col += (remaining_k_col % block_size);
            start_offset = local_k_index_col + 1;
        }
        if (remaining_k_row / block_size > executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k_row / block_size == executionSettings->programSettings->torus_row) {
            local_k_index_row += (remaining_k_row % block_size);
        }

        int row_diagonal_rank = (k / block_size) % executionSettings->programSettings->torus_height;
        int col_diagonal_rank = (k / block_size) % executionSettings->programSettings->torus_width;
        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_width, 0.0);
        if (row_diagonal_rank == executionSettings->programSettings->torus_row) {
            HOST_DATA_TYPE current_k;
            current_k = (local_k_index_col < matrix_width) ? b_tmp[local_k_index_col] : 0.0;
            MPI_Bcast(&current_k, 1, MPI_DATA_TYPE,  col_diagonal_rank, row_communicator);
            // For each row below add
            for (int i = start_offset; i < matrix_width; i++) {
                // add solved upper row to current row
                tmp_scaled_b[i] = current_k * data.A[matrix_width * local_k_index_row + i];
            }
        }
        MPI_Bcast(&tmp_scaled_b.data()[start_offset], matrix_width - start_offset, MPI_DATA_TYPE, row_diagonal_rank, col_communicator);
        for (int i = start_offset; i < matrix_width; i++) {
            // add solved upper row to current row
            b_tmp[i] += tmp_scaled_b[i];
        }
    }

    // now solve  u*x = y
    for (int k = global_matrix_size - 1; k >= 0; k--) {
        size_t local_k_index_col =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * executionSettings->programSettings->torus_height) * block_size;
        size_t remaining_k_col = k % (block_size * executionSettings->programSettings->torus_width);
        size_t remaining_k_row = k % (block_size * executionSettings->programSettings->torus_height);
        if (remaining_k_col / block_size > executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
        }
        else if (remaining_k_col / block_size == executionSettings->programSettings->torus_col) {
            local_k_index_col += remaining_k_col % block_size;
        }
        if (remaining_k_row / block_size > executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k_row / block_size == executionSettings->programSettings->torus_row) {
            local_k_index_row += remaining_k_row % block_size;
        }

        HOST_DATA_TYPE scale_element = (local_k_index_col < matrix_width && local_k_index_row < matrix_height) ? b_tmp[local_k_index_col] * data.A[matrix_width * local_k_index_row + local_k_index_col] : 0.0;
        int row_diagonal_rank = (k / block_size) % executionSettings->programSettings->torus_height;
        int col_diagonal_rank = (k / block_size) % executionSettings->programSettings->torus_width;
        MPI_Bcast(&scale_element, 1, MPI_DATA_TYPE, row_diagonal_rank, col_communicator);
        if (col_diagonal_rank == executionSettings->programSettings->torus_col) {
            b_tmp[local_k_index_col] = -scale_element;
        }
        MPI_Bcast(&scale_element, 1, MPI_DATA_TYPE, col_diagonal_rank, row_communicator);
        size_t end_offset = local_k_index_col;

        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_width, 0.0);
        if (row_diagonal_rank == executionSettings->programSettings->torus_row) {
            // For each row below add
            for (int i = 0; i < end_offset; i++) {
                tmp_scaled_b[i] = scale_element * data.A[matrix_width * local_k_index_row + i];
            }
        }
        MPI_Bcast(tmp_scaled_b.data(), end_offset, MPI_DATA_TYPE, row_diagonal_rank, col_communicator);
        for (int i = 0; i < end_offset; i++) {
            // add solved upper row to current row
            b_tmp[i] += tmp_scaled_b[i];
        }
    }
    for (int k = 0; k < b_tmp.size(); k++) {
        data.b[k] = b_tmp[k];
    }

#ifndef NDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < mpi_comm_size; rank++) {
        if (rank == mpi_comm_rank) {
            double sum = 0;
            double max = 0;
            for (int k = 0; k < matrix_width; k++) {
                sum += std::abs(data.b[k]);
                if (std::abs(data.b[k] - 1) > 0.1 || data.b[k] == NAN) {
                    std::cout << "Rank " << mpi_comm_rank << " Pos: " << k << " Value: " << std::abs(data.b[k]) << std::endl;
                }
            }
            std::cout << "Rank " << mpi_comm_rank << " Dist.Sum: " << sum << " Max: " << max << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
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
