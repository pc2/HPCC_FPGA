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

#ifndef SRC_HOST_LINPACK_BENCHMARK_H_
#define SRC_HOST_LINPACK_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>
#include <random>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "execution_types/execution_types.hpp"
#include "parameters.h"
#include "linpack_data.hpp"
extern "C" {
    #include "gmres.h"
}

/**
 * @brief Contains all classes and methods needed by the LINPACK benchmark
 * 
 */
namespace linpack {

/**
 * @brief Implementation of the Linpack benchmark
 * 
 */
template<class TDevice, class TContext, class TProgram>
class LinpackBenchmark : public hpcc_base::HpccFpgaBenchmark<LinpackProgramSettings, TDevice, TContext, TProgram, LinpackData, LinpackExecutionTimings> {

protected:

    /**
     * @brief Additional input parameters of the Linpack benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override {
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


    /**
     * @brief Distributed solving of l*y=b and u*x = y 
     * 
     * @param data The local data. b will contain the solution for the unknows that were handeled by this rank
     */
    void 
    distributed_gesl_nopvt_ref(linpack::LinpackData& data) {
    uint global_matrix_size = this->executionSettings->programSettings->matrixSize;
    uint matrix_width = data.matrix_width;
    uint matrix_height = data.matrix_height;
    uint block_size = this->executionSettings->programSettings->blockSize;
    // create a communicator to exchange the rows
    MPI_Comm row_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, this->executionSettings->programSettings->torus_row, 0,&row_communicator);
    MPI_Comm col_communicator;
    MPI_Comm_split(MPI_COMM_WORLD, this->executionSettings->programSettings->torus_col, 0,&col_communicator);
    std::vector<HOST_DATA_TYPE> b_tmp(matrix_width);

    for (int k = 0; k < b_tmp.size(); k++) {
        b_tmp[k] = data.b[k];
    }

    // solve l*y = b
    // For each row in matrix
    for (int k = 0; k < global_matrix_size - 1; k++) {
        size_t local_k_index_col =  k / (block_size * this->executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * this->executionSettings->programSettings->torus_height) * block_size;
        size_t remaining_k_col = k % (block_size * this->executionSettings->programSettings->torus_width);
        size_t remaining_k_row = k % (block_size * this->executionSettings->programSettings->torus_height);
        size_t start_offset = local_k_index_col;
        if (remaining_k_col / block_size > this->executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
            start_offset = local_k_index_col;
        }
        else if (remaining_k_col / block_size == this->executionSettings->programSettings->torus_col) {
            local_k_index_col += (remaining_k_col % block_size);
            start_offset = local_k_index_col + 1;
        }
        if (remaining_k_row / block_size > this->executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k_row / block_size == this->executionSettings->programSettings->torus_row) {
            local_k_index_row += (remaining_k_row % block_size);
        }

        int row_diagonal_rank = (k / block_size) % this->executionSettings->programSettings->torus_height;
        int col_diagonal_rank = (k / block_size) % this->executionSettings->programSettings->torus_width;
        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_width, 0.0);
        if (row_diagonal_rank == this->executionSettings->programSettings->torus_row) {
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
        size_t local_k_index_col =  k / (block_size * this->executionSettings->programSettings->torus_width) * block_size;
        size_t local_k_index_row =  k / (block_size * this->executionSettings->programSettings->torus_height) * block_size;
        size_t remaining_k_col = k % (block_size * this->executionSettings->programSettings->torus_width);
        size_t remaining_k_row = k % (block_size * this->executionSettings->programSettings->torus_height);
        if (remaining_k_col / block_size > this->executionSettings->programSettings->torus_col){
            local_k_index_col += block_size;
        }
        else if (remaining_k_col / block_size == this->executionSettings->programSettings->torus_col) {
            local_k_index_col += remaining_k_col % block_size;
        }
        if (remaining_k_row / block_size > this->executionSettings->programSettings->torus_row){
            local_k_index_row += block_size;
        }
        else if (remaining_k_row / block_size == this->executionSettings->programSettings->torus_row) {
            local_k_index_row += remaining_k_row % block_size;
        }

        HOST_DATA_TYPE scale_element = (local_k_index_col < matrix_width && local_k_index_row < matrix_height) ? b_tmp[local_k_index_col] * data.A[matrix_width * local_k_index_row + local_k_index_col] : 0.0;
        int row_diagonal_rank = (k / block_size) % this->executionSettings->programSettings->torus_height;
        int col_diagonal_rank = (k / block_size) % this->executionSettings->programSettings->torus_width;
        MPI_Bcast(&scale_element, 1, MPI_DATA_TYPE, row_diagonal_rank, col_communicator);
        if (col_diagonal_rank == this->executionSettings->programSettings->torus_col) {
            b_tmp[local_k_index_col] = -scale_element;
        }
        MPI_Bcast(&scale_element, 1, MPI_DATA_TYPE, col_diagonal_rank, row_communicator);
        size_t end_offset = local_k_index_col;

        std::vector<HOST_DATA_TYPE> tmp_scaled_b(matrix_width, 0.0);
        if (row_diagonal_rank == this->executionSettings->programSettings->torus_row) {
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
    for (int rank = 0; rank < this->mpi_comm_size; rank++) {
        if (rank == this->mpi_comm_rank) {
            double sum = 0;
            double max = 0;
            for (int k = 0; k < matrix_width; k++) {
                sum += std::abs(data.b[k]);
                if (std::abs(data.b[k] - 1) > 0.1 || data.b[k] == NAN) {
                    std::cout << "Rank " << this->mpi_comm_rank << " Pos: " << k << " Value: " << std::abs(data.b[k]) << std::endl;
                }
            }
            std::cout << "Rank " << this->mpi_comm_rank << " Dist.Sum: " << sum << " Max: " << max << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}


public:

    /**
     * @brief LINPACK specific implementation of the data generation
     * 
     * @return std::unique_ptr<LinpackData> The input and output data of the benchmark
     */
    std::unique_ptr<LinpackData>
    generateInputData() override {
    int local_matrix_width = this->executionSettings->programSettings->matrixSize / this->executionSettings->programSettings->torus_width;
    int local_matrix_height = this->executionSettings->programSettings->matrixSize / this->executionSettings->programSettings->torus_height;

    if ((this->executionSettings->programSettings->matrixSize / this->executionSettings->programSettings->blockSize) % this->executionSettings->programSettings->torus_width > 0 || 
        (this->executionSettings->programSettings->matrixSize / this->executionSettings->programSettings->blockSize) % this->executionSettings->programSettings->torus_height > 0) {
            throw std::runtime_error("Global matrix size must be multiple of LCM of PQ grid!");
    }

    auto d = std::unique_ptr<linpack::LinpackData>(new linpack::LinpackData(*this->executionSettings->context ,local_matrix_width, local_matrix_height));
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
    if (this->executionSettings->programSettings->isDiagonallyDominant) {
        // create a communicator to exchange the rows
        MPI_Comm row_communicator;
        MPI_Comm_split(MPI_COMM_WORLD, this->executionSettings->programSettings->torus_row, 0,&row_communicator);

        // Caclulate the sum for every row and insert in into the matrix
        for (int local_matrix_row = 0; local_matrix_row < local_matrix_height; local_matrix_row++) {
            int blockSize = this->executionSettings->programSettings->blockSize;
            int global_matrix_row = this->executionSettings->programSettings->torus_row * blockSize + (local_matrix_row / blockSize) * blockSize * this->executionSettings->programSettings->torus_height + (local_matrix_row % blockSize);
            int local_matrix_col = (global_matrix_row - this->executionSettings->programSettings->torus_col * blockSize) / (blockSize * this->executionSettings->programSettings->torus_width) * blockSize + (global_matrix_row % blockSize);
            int diagonal_rank = (global_matrix_row / blockSize) % this->executionSettings->programSettings->torus_width;
            bool diagonal_on_this_rank = diagonal_rank == this->executionSettings->programSettings->torus_col;
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
    MPI_Comm_split(MPI_COMM_WORLD, this->executionSettings->programSettings->torus_col, 0,&col_communicator);

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


    /**
     * @brief Linpack specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<LinpackExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<LinpackExecutionTimings>
    executeKernel(LinpackData &data) override {
    std::unique_ptr<linpack::LinpackExecutionTimings> timings;
    switch (this->executionSettings->programSettings->communicationType) {
#ifdef USE_OCL_HOST
        case hpcc_base::CommunicationType::pcie_mpi : timings = execution::pcie::calculate(*this->executionSettings, data); break;
        case hpcc_base::CommunicationType::intel_external_channels: timings = execution::iec::calculate(*this->executionSettings, data); break;
#endif
#ifdef USE_XRT_HOST
        case hpcc_base::CommunicationType::pcie_mpi : timings = execution::xrt_pcie::calculate(*this->executionSettings, data); break;
        case hpcc_base::CommunicationType::accl : timings = execution::accl_buffers::calculate(*this->executionSettings, data); break;
#endif
        default: throw std::runtime_error("No calculate method implemented for communication type " + commToString(this->executionSettings->programSettings->communicationType));
    }
#ifdef DISTRIBUTED_VALIDATION
    distributed_gesl_nopvt_ref(data);
#endif
    return timings;
}


    /**
     * @brief Linpack specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true If validation is successful
     * @return false otherwise
     */
    bool
    validateOutputAndPrintError(LinpackData &data) override {
    uint n= this->executionSettings->programSettings->matrixSize;
    uint matrix_width = data.matrix_width;
    uint matrix_height = data.matrix_height;
    double residn;
    double resid = 0.0;
    double normx = 0.0;
#ifndef DISTRIBUTED_VALIDATION
    if (mpi_comm_rank > 0) {
        for (int j = 0; j < matrix_height; j++) {
            for (int i = 0; i < matrix_width; i+= this->executionSettings->programSettings->blockSize) {
                MPI_Send(&data.A[matrix_width * j + i], this->executionSettings->programSettings->blockSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
            }
        }
        if (executionSettings->programSettings->torus_row == 0) {
            for (int i = 0; i < matrix_width; i+= this->executionSettings->programSettings->blockSize) {
                MPI_Send(&data.b[i], this->executionSettings->programSettings->blockSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
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
            for (int i = 0; i < n; i+= this->executionSettings->programSettings->blockSize) {
                int recvcol= (i / this->executionSettings->programSettings->blockSize) % this->executionSettings->programSettings->torus_width;
                int recvrow= (j / this->executionSettings->programSettings->blockSize) % this->executionSettings->programSettings->torus_height;
                int recvrank = this->executionSettings->programSettings->torus_width * recvrow + recvcol;
                if (recvrank > 0) {
                    MPI_Recv(&total_a[j * n + i],executionSettings->programSettings->blockSize, MPI_DATA_TYPE, recvrank, 0, MPI_COMM_WORLD,  &status);
                }
                else {
                    for (int k=0; k < this->executionSettings->programSettings->blockSize; k++) {
                        total_a[j * n + i + k] = data.A[current_offset + k];
                    }
                    current_offset += this->executionSettings->programSettings->blockSize;
                }
            }
        }
        current_offset = 0;
        for (int i = 0; i < n; i+= this->executionSettings->programSettings->blockSize) {
            int recvcol= (i / this->executionSettings->programSettings->blockSize) % this->executionSettings->programSettings->torus_width;
            if (recvcol > 0) {
                MPI_Recv(&total_b[i], this->executionSettings->programSettings->blockSize, MPI_DATA_TYPE, recvcol, 0, MPI_COMM_WORLD, &status);
            }
            else {
                for (int k=0; k < this->executionSettings->programSettings->blockSize; k++) {
                    total_b[i + k] = data.b[current_offset + k];
                }
                current_offset += this->executionSettings->programSettings->blockSize;
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
    std::cout << "Rank " << this->mpi_comm_rank << ": resid=" << local_resid << ", normx=" << local_normx << std::endl;
#endif

    MPI_Reduce(&local_resid, &resid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_normx, &normx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#endif


    HOST_DATA_TYPE eps = std::numeric_limits<HOST_DATA_TYPE>::epsilon();
    residn = resid / (static_cast<double>(n)*normx*eps);

    #ifndef NDEBUG
        if (residn > 1 &&  this->mpi_comm_size == 1) {
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
            if (this->executionSettings->programSettings->isDiagonallyDominant) {
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

    if (this->mpi_comm_rank == 0) {
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


    /**
     * @brief Linpack specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    collectAndPrintResults(const LinpackExecutionTimings &output) override {
    // Calculate performance for kernel execution plus data transfer
    double tmean = 0;
    double tlumean = 0;
    double tslmean = 0;
    double tmin = std::numeric_limits<double>::max();
    double lu_min = std::numeric_limits<double>::max();
    double sl_min = std::numeric_limits<double>::max();

#ifndef NDEBUG
    std::cout << "Rank " << this->mpi_comm_rank << ": Result collection started" << std::endl;
#endif

    std::vector<double> global_lu_times(output.gefaTimings.size());
    MPI_Reduce(output.gefaTimings.data(), global_lu_times.data(), output.gefaTimings.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> global_sl_times(output.geslTimings.size());
    MPI_Reduce(output.geslTimings.data(), global_sl_times.data(), output.geslTimings.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#ifndef NDEBUG
    std::cout << "Rank " << this->mpi_comm_rank << ": Result collection done" << std::endl;
#endif


    if (this->mpi_comm_rank > 0) {
        // Only the master rank needs to calculate and print result
        return;
    }

    double total_matrix_size = static_cast<double>(this->executionSettings->programSettings->matrixSize);
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

    /**
     * @brief Construct a new Linpack Benchmark object
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    LinpackBenchmark(int argc, char* argv[]) : hpcc_base::HpccFpgaBenchmark<linpack::LinpackProgramSettings, TDevice, TContext, TProgram, linpack::LinpackData, linpack::LinpackExecutionTimings>(argc, argv) {
        this->setupBenchmark(argc, argv);
    }

        /**
     * @brief Construct a new Linpack Benchmark object
     */
    LinpackBenchmark();

};

} // namespace linpack


#endif // SRC_HOST_STREAM_BENCHMARK_H_
