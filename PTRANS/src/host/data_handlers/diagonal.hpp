/*
Copyright (c) 2020 Marius Meyer

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

#ifndef SRC_HOST_TRANSPOSE_HANDLER_DIAGONAL_HPP_
#define SRC_HOST_TRANSPOSE_HANDLER_DIAGONAL_HPP_

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "handler.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {

    namespace data_handler {

/**
 * @brief Transposes the data over external channels, so every part of a pair is located on a different FPGA.
 *         Data will be distributed to the ranks such that only a fixed pair of ranks will communicate to exchange
 *         the missing data. e.g. for N ranks, the pairs will be (0, N/2), (1, N/2 + 1), ...
 * 
 */
template<class TDevice, class TContext, class TProgram>
class DistributedDiagonalTransposeDataHandler : public TransposeDataHandler<TDevice, TContext, TProgram> {

private:

    /**
     * @brief Number of diagonal ranks that will sent the blcoks to themselves
     * 
     */
    int num_diagonal_ranks;

    /**
     * @brief MPI data for matrix blocks
     * 
     */
    MPI_Datatype data_block;

public:

    /**
     * @brief Generate data for transposition based on the implemented distribution scheme
     * 
     * @param settings The execution settings that contain information about the data size
     * @return std::unique_ptr<TransposeData> The generated data
     */
    std::unique_ptr<TransposeData<TContext>>
    generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings, TDevice, TContext, TProgram>& settings) override {
        MPI_Type_contiguous(settings.programSettings->blockSize * settings.programSettings->blockSize, MPI_FLOAT, &data_block);
        MPI_Type_commit(&data_block);
        
        int width_in_blocks = settings.programSettings->matrixSize / settings.programSettings->blockSize;

        int avg_blocks_per_rank = (width_in_blocks * width_in_blocks) / this->mpi_comm_size;
        int avg_diagonal_blocks = width_in_blocks;
        if (avg_blocks_per_rank > 0) {
            avg_diagonal_blocks = (width_in_blocks / avg_blocks_per_rank);
        }
        num_diagonal_ranks = std::max(avg_diagonal_blocks, 1);

        if (num_diagonal_ranks % 2 != this->mpi_comm_size % 2) {
        #ifndef NDEBUG
            std::cout << "Rank " << this->mpi_comm_rank << ": Fail 1!" << std::endl;
        #endif
            // Abort if there is a too high difference in the number of matrix blocks between the MPI ranks
            throw std::runtime_error("Matrix size and MPI ranks to not allow fair distribution of blocks! Increase or reduce the number of MPI ranks by 1.");
        }
        if ((this->mpi_comm_size - num_diagonal_ranks) % 2 != 0 || (this->mpi_comm_size - num_diagonal_ranks) == 0 && width_in_blocks > 1) {
        #ifndef NDEBUG
            std::cout << "Rank " << this->mpi_comm_rank << ": Fail 2!" << std::endl;
        #endif
            throw std::runtime_error("Not possible to create pairs of MPI ranks for lower and upper half of matrix. Increase number of MPI ranks!.");
        }
        bool this_rank_is_diagonal = this->mpi_comm_rank >= (this->mpi_comm_size - num_diagonal_ranks);
        int blocks_if_diagonal = width_in_blocks / num_diagonal_ranks + ( (this->mpi_comm_rank - (this->mpi_comm_size - num_diagonal_ranks)) < (width_in_blocks % num_diagonal_ranks) ? 1 : 0);
        int blocks_if_not_diagonal = 0;
        if ((this->mpi_comm_size - num_diagonal_ranks) > 0 ) {
            blocks_if_not_diagonal = (width_in_blocks * (width_in_blocks - 1)) / (this->mpi_comm_size - num_diagonal_ranks) + (this->mpi_comm_rank  < ((width_in_blocks * (width_in_blocks - 1)) % (this->mpi_comm_size - num_diagonal_ranks)) ? 1 : 0);
        }


        int blocks_per_rank = (this_rank_is_diagonal) ? blocks_if_diagonal : blocks_if_not_diagonal;

        if (this->mpi_comm_rank == 0) {
            std::cout << "Diag. blocks per rank:              " << blocks_if_diagonal << std::endl;
            std::cout << "Blocks per rank:                    " << blocks_if_not_diagonal << std::endl;
            std::cout << "Loopback ranks for diagonal blocks: " << num_diagonal_ranks << std::endl;
        }
        // Height of a matrix generated for a single memory bank on a single MPI rank
        int data_height_per_rank = blocks_per_rank * settings.programSettings->blockSize;

    #ifndef NDEBUG
        std::cout << "Rank " << this->mpi_comm_rank << ": NumBlocks = " << blocks_per_rank << std::endl;
    #endif
        
        // Allocate memory for a single device and all its memory banks
        auto d = std::unique_ptr<transpose::TransposeData<TContext>>(new transpose::TransposeData<TContext>(*settings.context, settings.programSettings->blockSize, blocks_per_rank));

        // Fill the allocated memory with pseudo random values
        std::mt19937 gen(this->mpi_comm_rank);
        std::uniform_real_distribution<> dis(-100.0, 100.0);
        for (size_t i = 0; i < data_height_per_rank; i++) {
            for (size_t j = 0; j < settings.programSettings->blockSize; j++) {
                d->A[i * settings.programSettings->blockSize + j] = dis(gen);
                d->B[i * settings.programSettings->blockSize + j] = dis(gen);
                d->result[i * settings.programSettings->blockSize + j] = 0.0;
            }
        }
        
        return d;
    }

    /**
     * @brief Exchange the data blocks for verification
     * 
     * @param data The data that was generated locally and will be exchanged with other MPI ranks 
     *              Exchanged data will be stored in the same object.
     */
    void
    exchangeData(TransposeData<TContext>& data) override {

    #ifndef NDEBUG
        // std::cout << "Start data exchange " << mpi_comm_rank << std::endl;
    #endif
        // Only need to exchange data, if rank has a partner
        if (this->mpi_comm_rank < this->mpi_comm_size - num_diagonal_ranks) {

            int first_upper_half_rank = (this->mpi_comm_size - num_diagonal_ranks)/2;
            int pair_rank = (this->mpi_comm_rank >= first_upper_half_rank) ? this->mpi_comm_rank - first_upper_half_rank : this->mpi_comm_rank + first_upper_half_rank;

            // To re-calculate the matrix transposition locally on this host, we need to 
            // exchange matrix A for every kernel replication
            // The order of the matrix blocks does not change during the exchange, because they are distributed diagonally 
            // and will be handled in the order below:
            //
            // . . 1 3
            // . . . 2
            // 1 . . .
            // 3 2 . .
            MPI_Status status;        

            size_t remaining_data_size = data.numBlocks;
            size_t offset = 0;
            while (remaining_data_size > 0) {
                int next_chunk = (remaining_data_size > std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max(): remaining_data_size;
                MPI_Sendrecv(&data.A[offset], next_chunk, data_block, pair_rank, 0, &data.exchange[offset], next_chunk, data_block, pair_rank, 0, MPI_COMM_WORLD, &status);

                remaining_data_size -= next_chunk;
                offset += static_cast<size_t>(next_chunk) * static_cast<size_t>(data.blockSize * data.blockSize);
            }
            
            // Exchange window pointers
            HOST_DATA_TYPE* tmp = data.exchange;
            data.exchange = data.A;
            data.A = tmp;
        }
    #ifndef NDEBUG
        // std::cout << "End data exchange " << mpi_comm_rank << std::endl;
    #endif
    }

    void 
    reference_transpose(TransposeData<TContext>& data) {
        size_t block_offset = data.blockSize * data.blockSize;
        for (size_t b = 0; b < data.numBlocks; b++) {
            for (size_t i = 0; i < data.blockSize; i++) {
                for (size_t j = 0; j < data.blockSize; j++) {
                    data.A[b * block_offset + j * data.blockSize + i] -= (data.result[b * block_offset + i * data.blockSize + j] 
                                                                                - data.B[b * block_offset + i * data.blockSize + j]);
                }
            }
        }
    }

    DistributedDiagonalTransposeDataHandler(int mpi_rank, int mpi_size): TransposeDataHandler<TDevice, TContext, TProgram>(mpi_rank, mpi_size) {
        if (mpi_rank >= mpi_size) {
            throw std::runtime_error("MPI rank must be smaller the MPI world size!");
        }
    }

};

} 
}

#endif
