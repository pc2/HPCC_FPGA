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
#include "execution.h"
#include "parameters.h"

linpack::LinpackProgramSettings::LinpackProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * (1 << (results["b"].as<uint>()))), blockSize(1 << (results["b"].as<uint>())), 
    isEmulationKernel(results.count("emulation") > 0), isDiagonallyDominant(results.count("uniform") == 0) {
    int mpi_comm_rank;
    int mpi_comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
    // calculate the row and column of the MPI rank in the torus 
    torus_row = (mpi_comm_rank / std::sqrt(mpi_comm_size));
    torus_col = (mpi_comm_rank % static_cast<int>(std::sqrt(mpi_comm_size)));
    torus_width = static_cast<int>(std::sqrt(mpi_comm_size));
}

std::map<std::string, std::string>
linpack::LinpackProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Emulate"] = (isEmulationKernel) ? "Yes" : "No";
        return map;
}

linpack::LinpackData::LinpackData(cl::Context context, size_t size) : norma(0.0), context(context) {
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
    posix_memalign(reinterpret_cast<void**>(&A), 4096, size * size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&b), 4096, size * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&ipvt), 4096, size * sizeof(cl_int));
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
    if (static_cast<int>(std::sqrt(mpi_comm_size) * std::sqrt(mpi_comm_size)) != mpi_comm_size) {
        throw std::runtime_error("ERROR: MPI communication size must be a square number!");
    }
}

void
linpack::LinpackBenchmark::addAdditionalParseOptions(cxxopts::Options &options) {
    options.add_options()
        ("m", "Matrix size in number of blocks in one dimension for a singe MPI rank. Total matrix will have size m * sqrt(MPI_size)",
            cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_MATRIX_SIZE)))
        ("b", "Log2 of the block size in number of values in one dimension",
            cxxopts::value<uint>()->default_value(std::to_string(LOCAL_MEM_BLOCK_LOG)))
        ("uniform", "Generate a uniform matrix instead of a diagonally dominant. This has to be supported by the FPGA kernel!")
        ("emulation", "Use kernel arguments for emulation. This may be necessary to simulate persistent local memory on the FPGA");
}

std::unique_ptr<linpack::LinpackExecutionTimings>
linpack::LinpackBenchmark::executeKernel(LinpackData &data) {
    auto timings = bm_execution::calculate(*executionSettings, data.A, data.b, data.ipvt);
#ifdef DISTRIBUTED_VALIDATION
    distributed_gesl_nopvt_ref(data);
#endif
    return timings;
}

void
linpack::LinpackBenchmark::collectAndPrintResults(const linpack::LinpackExecutionTimings &output) {
    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tlumean = 0;
    double tslmean = 0;
    double tmin = std::numeric_limits<double>::max();
    double lu_min = std::numeric_limits<double>::max();
    double sl_min = std::numeric_limits<double>::max();

#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": Result collection started" << std::endl;
#endif

    std::vector<double> global_lu_times(output.gefaTimings.size());
    MPI_Reduce(output.gefaTimings.data(), global_lu_times.data(), output.gefaTimings.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> global_sl_times(output.geslTimings.size());
    MPI_Reduce(output.geslTimings.data(), global_sl_times.data(), output.geslTimings.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": Result collection done" << std::endl;
#endif


    if (mpi_comm_rank > 0) {
        // Only the master rank needs to calculate and print result
        return;
    }

    double total_matrix_size = static_cast<double>(executionSettings->programSettings->matrixSize) * executionSettings->programSettings->torus_width;
    double gflops_lu = ((2.0e0*total_matrix_size * total_matrix_size * total_matrix_size)/ 3.0) / 1.0e9; 
    double gflops_sl = (2.0*(total_matrix_size * total_matrix_size))/1.0e9;
    for (int i =0; i < global_lu_times.size(); i++) {
        double currentTime = global_lu_times[i] + global_sl_times[i];
        tmean +=  currentTime;
        tlumean +=  global_lu_times[i];
        tslmean += global_sl_times[i];
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
    tmean = tmean / global_lu_times.size();
    tlumean = tlumean / global_lu_times.size();
    tslmean = tslmean / global_sl_times.size();

     std::cout << std::setw(ENTRY_SPACE)
              << "Method" << std::setw(ENTRY_SPACE)
              << "best" << std::setw(ENTRY_SPACE) << "mean"
              << std::setw(ENTRY_SPACE) << "GFLOPS" << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "total" << std::setw(ENTRY_SPACE)
              << tmin << std::setw(ENTRY_SPACE) << tmean
              << std::setw(ENTRY_SPACE) << ((gflops_lu + gflops_sl) / tmin)
              << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "GEFA" << std::setw(ENTRY_SPACE)
            << lu_min << std::setw(ENTRY_SPACE) << tlumean
            << std::setw(ENTRY_SPACE) << ((gflops_lu) / lu_min)
            << std::endl;

    std::cout << std::setw(ENTRY_SPACE) << "GESL" << std::setw(ENTRY_SPACE)
              << sl_min << std::setw(ENTRY_SPACE) << tslmean
              << std::setw(ENTRY_SPACE) << (gflops_sl / sl_min)
              << std::endl;
}

std::unique_ptr<linpack::LinpackData>
linpack::LinpackBenchmark::generateInputData() {
    auto d = std::unique_ptr<linpack::LinpackData>(new linpack::LinpackData(*executionSettings->context ,executionSettings->programSettings->matrixSize));
    std::mt19937 gen(this->mpi_comm_rank);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    d->norma = 0.0;
    /*
    Generate a matrix by using pseudo random number in the range (0,1)
    */
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        // fill a single column of the matrix
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
                HOST_DATA_TYPE temp = dis(gen);
                d->A[executionSettings->programSettings->matrixSize*j+i] = dis(gen);
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
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            // set the diagonal elements of the matrix to 0
            if (executionSettings->programSettings->torus_row == executionSettings->programSettings->torus_col) {
                d->A[executionSettings->programSettings->matrixSize*j + j] = 0.0;
            }
            HOST_DATA_TYPE local_row_sum = 0.0;
            for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
                local_row_sum += d->A[executionSettings->programSettings->matrixSize*j + i];
            } 
            HOST_DATA_TYPE row_sum = 0.0;
            MPI_Reduce(&local_row_sum, &row_sum, 1, MPI_FLOAT, MPI_SUM, executionSettings->programSettings->torus_row, row_communicator);
            // insert row sum into matrix if it contains the diagonal block
            if (executionSettings->programSettings->torus_row == executionSettings->programSettings->torus_col) {
                // update norm of local matrix
                d->norma = (row_sum > d->norma) ? row_sum : d->norma;
                d->A[executionSettings->programSettings->matrixSize*j + j] = row_sum;
            }
        }
    }
        
    // initialize other vectors
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        d->b[i] = 0.0;
        d->ipvt[i] = i;
    }

    MPI_Comm col_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_col, 0,&col_communicator);

    // Generate vector b by accumulating the columns of the matrix.
    // This will lead to a result vector x with ones on every position
    // Every rank will have a valid part of the final b vector stored
    for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
        HOST_DATA_TYPE local_col_sum = 0.0;
        for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
            local_col_sum += d->A[executionSettings->programSettings->matrixSize*i+j];
        }
        HOST_DATA_TYPE row_sum = 0.0;
        MPI_Allreduce(&local_col_sum, &(d->b[j]), 1, MPI_FLOAT, MPI_SUM, col_communicator);      
    }
    return d;
}

bool  
linpack::LinpackBenchmark::validateOutputAndPrintError(linpack::LinpackData &data) {
    uint n= executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width;
    double residn;
    double resid = 0.0;
    double normx = 0.0;
#ifndef DISTRIBUTED_VALIDATION
    if (mpi_comm_rank > 0) {
        for (int j = 0; j < executionSettings->programSettings->matrixSize; j++) {
            for (int i = 0; i < executionSettings->programSettings->matrixSize; i+= executionSettings->programSettings->blockSize) {
                MPI_Send(&data.A[executionSettings->programSettings->matrixSize * j + i], executionSettings->programSettings->blockSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
        }
        if (executionSettings->programSettings->torus_row == 0) {
            for (int i = 0; i < executionSettings->programSettings->matrixSize; i+= executionSettings->programSettings->blockSize) {
                MPI_Send(&data.b[i], executionSettings->programSettings->blockSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }
        }
        residn = 0;
    }
    else {
        MPI_Status status;
        size_t current_offset = 0;
        std::vector<HOST_DATA_TYPE> total_b_original(executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width);
        std::vector<HOST_DATA_TYPE> total_b(executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width);
        std::vector<HOST_DATA_TYPE> total_a(executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width*executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width);
        for (int j = 0; j < executionSettings->programSettings->matrixSize* executionSettings->programSettings->torus_width; j++) {
            for (int i = 0; i < executionSettings->programSettings->matrixSize* executionSettings->programSettings->torus_width; i+= executionSettings->programSettings->blockSize) {
                int recvcol= (i / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width;
                int recvrow= (j / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width;
                int recvrank = executionSettings->programSettings->torus_width * recvrow + recvcol;
                if (recvrank > 0) {
                    MPI_Recv(&total_a[j * executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width + i],executionSettings->programSettings->blockSize, MPI_FLOAT, recvrank, 0, MPI_COMM_WORLD,  &status);
                }
                else {
                    for (int k=0; k < executionSettings->programSettings->blockSize; k++) {
                        total_a[j * executionSettings->programSettings->matrixSize * executionSettings->programSettings->torus_width + i + k] = data.A[current_offset + k];
                    }
                    current_offset += executionSettings->programSettings->blockSize;
                }
            }
        }
        current_offset = 0;
        for (int i = 0; i < executionSettings->programSettings->matrixSize* executionSettings->programSettings->torus_width; i+= executionSettings->programSettings->blockSize) {
            int recvcol= (i / executionSettings->programSettings->blockSize) % executionSettings->programSettings->torus_width;
            if (recvcol > 0) {
                MPI_Recv(&total_b[i], executionSettings->programSettings->blockSize, MPI_FLOAT, recvcol, 0, MPI_COMM_WORLD, &status);
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
    auto unchanged_data = generateInputData();
    double local_resid = 0;
    double local_normx = 0;
    for (int i = 0; i < executionSettings->programSettings->matrixSize; i++) {
        local_resid = (local_resid > std::abs(data.b[i] - 1)) ? local_resid : std::abs(data.b[i] - 1);
        local_normx = (local_normx > std::abs(unchanged_data->b[i])) ? local_normx : std::abs(unchanged_data->b[i]);
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
                    std::cout << ref_result->A[n * j + i] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    #endif

    if (mpi_comm_rank == 0) {
        //std::cout << resid << ", " << norma << ", " << normx << std::endl;
        std::cout << "  norm. resid        resid       "\
                    "machep   " << std::endl;
        std::cout << std::setw(ENTRY_SPACE) << residn << std::setw(ENTRY_SPACE)
                << resid << std::setw(ENTRY_SPACE) << eps << std::endl;
        return residn < 1;
    }
    else {
        return true;
    }
}

void 
linpack::LinpackBenchmark::distributed_gesl_nopvt_ref(linpack::LinpackData& data) {
    uint matrix_size = executionSettings->programSettings->matrixSize;
    uint block_size = executionSettings->programSettings->blockSize;
    uint total_matrix_size = matrix_size * block_size;
    // create a communicator to exchange the rows
    MPI_Comm row_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_row, 0,&row_communicator);
    MPI_Comm col_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, executionSettings->programSettings->torus_col, 0,&col_communicator);
    std::vector<HOST_DATA_TYPE> b_tmp(matrix_size);
    // if 0= diagonal rank, if negative= lower rank, if positive = upper rank
    int op_mode = executionSettings->programSettings->torus_col - executionSettings->programSettings->torus_row;


    for (int k = 0; k < matrix_size; k++) {
        b_tmp[k] = data.b[k];
    }

    // solve l*y = b
    // For each row in matrix
    for (int k = 0; k < matrix_size * executionSettings->programSettings->torus_width - 1; k++) {
        size_t local_k_index_col =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t remaining_k = k % (block_size * executionSettings->programSettings->torus_width);
        size_t start_offset = local_k_index_col;
        if (remaining_k / block_size > executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
            start_offset = local_k_index_col;
        }
        else if (remaining_k / block_size == executionSettings->programSettings->torus_col) {
            local_k_index_col += (remaining_k % block_size);
            start_offset = local_k_index_col + 1;
        }
        if (remaining_k / block_size > executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k / block_size == executionSettings->programSettings->torus_row) {
            local_k_index_row += (remaining_k % block_size);
        }

        int current_bcast = (k / block_size) % executionSettings->programSettings->torus_width;
        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_size, 0.0);
        if ((k / block_size) % executionSettings->programSettings->torus_width == executionSettings->programSettings->torus_row) {
            HOST_DATA_TYPE current_k;
            current_k = (local_k_index_col < matrix_size) ? b_tmp[local_k_index_col] : 0.0;
            MPI_Bcast(&current_k, 1, MPI_FLOAT,  current_bcast, row_communicator);
            // For each row below add
            for (int i = start_offset; i < matrix_size; i++) {
                // add solved upper row to current row
                tmp_scaled_b[i] = current_k * data.A[matrix_size * local_k_index_row + i];
            }
        }
        MPI_Bcast(tmp_scaled_b.data(), matrix_size, MPI_FLOAT, current_bcast, col_communicator);
        for (int i = 0; i < matrix_size; i++) {
            // add solved upper row to current row
            b_tmp[i] += tmp_scaled_b[i];
        }
    }

    // now solve  u*x = y
    for (int k = matrix_size * executionSettings->programSettings->torus_width - 1; k >= 0; k--) {
        size_t local_k_index_col =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * executionSettings->programSettings->torus_width) * block_size;
        size_t remaining_k = k % (block_size * executionSettings->programSettings->torus_width);
        if (remaining_k / block_size > executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
        }
        else if (remaining_k / block_size == executionSettings->programSettings->torus_col) {
            local_k_index_col += remaining_k % block_size;
        }
        if (remaining_k / block_size > executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k / block_size == executionSettings->programSettings->torus_row) {
            local_k_index_row += remaining_k % block_size;
        }

        HOST_DATA_TYPE scale_element = (local_k_index_col < matrix_size && local_k_index_row < matrix_size) ? b_tmp[local_k_index_col] * data.A[matrix_size * local_k_index_row + local_k_index_col] : 0.0;
        MPI_Bcast(&scale_element, 1, MPI_FLOAT, executionSettings->programSettings->torus_col, col_communicator);
        if ((k / block_size) % executionSettings->programSettings->torus_width == executionSettings->programSettings->torus_col) {
            b_tmp[local_k_index_col] = -scale_element;
        }
        MPI_Bcast(&scale_element, 1, MPI_FLOAT, executionSettings->programSettings->torus_row, row_communicator);
        size_t end_offset = local_k_index_col;

        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_size, 0.0);
        if ((k / block_size) % executionSettings->programSettings->torus_width == executionSettings->programSettings->torus_row) {
            // For each row below add
            for (int i = 0; i < end_offset; i++) {
                tmp_scaled_b[i] = scale_element * data.A[matrix_size * local_k_index_row + i];
            }
        }
        int current_bcast = (k / block_size) % executionSettings->programSettings->torus_width;
        MPI_Bcast(tmp_scaled_b.data(), matrix_size, MPI_FLOAT, current_bcast, col_communicator);
        for (int i = 0; i < matrix_size; i++) {
            // add solved upper row to current row
            b_tmp[i] += tmp_scaled_b[i];
        }
    }
    for (int k = 0; k < matrix_size; k++) {
        data.b[k] = b_tmp[k];
    }
#ifndef NDEBUG
    double sum = 0;
    double max = 0;
    for (int k = 0; k < matrix_size; k++) {
        sum += std::abs(data.b[k]);
        if (std::abs(data.b[k] - 1) > 0.1) {
            std::cout << "Rank " << mpi_comm_rank << " Pos: " << k << " Value: " << std::abs(data.b[k]) << std::endl;
        }
    }

    std::cout << "Rank " << mpi_comm_rank << " Dist.Sum: " << sum << " Max: " << max << std::endl;
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
