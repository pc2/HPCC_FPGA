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

#include <random>

#ifdef _USE_MPI_
#include "mpi.h"
#endif

#include "transpose_handlers.hpp"

// Add every data handler that should be selectable from the command line into this map
// and also specify a string identifier for it
std::map<std::string, std::unique_ptr<transpose::TransposeDataHandler> (*)(int rank, int size)> transpose::dataHandlerIdentifierMap{
        // distributed external data handler
#ifdef _USE_MPI_
        {TRANSPOSE_HANDLERS_DIST_DIAG, &generateDataHandler<transpose::DistributedDiagonalTransposeDataHandler>}
#endif
        };

#ifdef _USE_MPI_

transpose::DistributedDiagonalTransposeDataHandler::DistributedDiagonalTransposeDataHandler(int rank, int size) : TransposeDataHandler(rank, size) {
    if (rank >= size) {
        throw std::runtime_error("MPI rank must be smaller the MPI world size!");
    }
}


std::unique_ptr<transpose::TransposeData> transpose::DistributedDiagonalTransposeDataHandler::generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) {
    int width_in_blocks = settings.programSettings->matrixSize / settings.programSettings->blockSize;

    int avg_blocks_per_rank = (width_in_blocks * width_in_blocks) / mpi_comm_size;
    int avg_diagonal_blocks = width_in_blocks;
    if (avg_blocks_per_rank > 0) {
        avg_diagonal_blocks = (width_in_blocks / avg_blocks_per_rank);
    }
    num_diagonal_ranks = std::max(avg_diagonal_blocks, 1);

    if (num_diagonal_ranks % 2 != mpi_comm_size % 2) {
    #ifndef NDEBUG
        std::cout << "Rank " << mpi_comm_rank << ": Fail 1!" << std::endl;
    #endif
        // Abort if there is a too high difference in the number of matrix blocks between the MPI ranks
        throw std::runtime_error("Matrix size and MPI ranks to not allow fair distribution of blocks! Increase or reduce the number of MPI ranks by 1.");
    }
    if ((mpi_comm_size - num_diagonal_ranks) % 2 != 0 || (mpi_comm_size - num_diagonal_ranks) == 0 && width_in_blocks > 1) {
    #ifndef NDEBUG
        std::cout << "Rank " << mpi_comm_rank << ": Fail 2!" << std::endl;
    #endif
        throw std::runtime_error("Not possible to create pairs of MPI ranks for lower and upper half of matrix. Increase number of MPI ranks!.");
    }
    bool this_rank_is_diagonal = mpi_comm_rank >= (mpi_comm_size - num_diagonal_ranks);
    int blocks_if_diagonal = width_in_blocks / num_diagonal_ranks + ( (mpi_comm_rank - (mpi_comm_size - num_diagonal_ranks)) < (width_in_blocks % num_diagonal_ranks) ? 1 : 0);
    int blocks_if_not_diagonal = 0;
    if ((mpi_comm_size - num_diagonal_ranks) > 0 ) {
        blocks_if_not_diagonal = (width_in_blocks * (width_in_blocks - 1)) / (mpi_comm_size - num_diagonal_ranks) + (mpi_comm_rank  < ((width_in_blocks * (width_in_blocks - 1)) % (mpi_comm_size - num_diagonal_ranks)) ? 1 : 0);
    }


    int blocks_per_rank = (this_rank_is_diagonal) ? blocks_if_diagonal : blocks_if_not_diagonal;

    if (mpi_comm_rank == 0) {
        std::cout << "Diag. blocks per rank:              " << blocks_if_diagonal << std::endl;
        std::cout << "Blocks per rank:                    " << blocks_if_not_diagonal << std::endl;
        std::cout << "Loopback ranks for diagonal blocks: " << num_diagonal_ranks << std::endl;
    }
    // Height of a matrix generated for a single memory bank on a single MPI rank
    int data_height_per_rank = blocks_per_rank * settings.programSettings->blockSize;

#ifndef NDEBUG
    std::cout << "Rank " << mpi_comm_rank << ": NumBlocks = " << blocks_per_rank << std::endl;
#endif
    
    // Allocate memory for a single device and all its memory banks
    auto d = std::unique_ptr<transpose::TransposeData>(new transpose::TransposeData(*settings.context, settings.programSettings->blockSize, blocks_per_rank));

    // Fill the allocated memory with pseudo random values
    std::mt19937 gen(mpi_comm_rank);
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

void transpose::DistributedDiagonalTransposeDataHandler::exchangeData(transpose::TransposeData& data) {
    // Only need to exchange data, if rank has a partner
    if (mpi_comm_rank < mpi_comm_size - num_diagonal_ranks) {
        int first_upper_half_rank = (mpi_comm_size - num_diagonal_ranks)/2;
        int pair_rank = (mpi_comm_rank >= first_upper_half_rank) ? mpi_comm_rank - first_upper_half_rank : mpi_comm_rank + first_upper_half_rank;

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
        size_t remaining_data_size = static_cast<size_t>(data.blockSize) * data.blockSize * data.numBlocks;
        size_t offset = 0;
        while (remaining_data_size > 0) {
            int next_chunk = (remaining_data_size > std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max(): remaining_data_size;
            MPI_Sendrecv_replace(&data.A[offset], next_chunk, MPI_FLOAT, pair_rank, 0, pair_rank, 0, MPI_COMM_WORLD, &status);
            remaining_data_size -= next_chunk;
            offset += next_chunk;
        }
    }
}

#endif
