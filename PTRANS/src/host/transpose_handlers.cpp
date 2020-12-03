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
        {TRANSPOSE_HANDLERS_DIST_EXT, &generateDataHandler<transpose::DistributedExternalTransposeDataHandler>}
#endif
        };

#ifdef _USE_MPI_

transpose::DistributedExternalTransposeDataHandler::DistributedExternalTransposeDataHandler(int rank, int size) : TransposeDataHandler(rank, size) {
    // Only works, if the number of MPI ranks is a multiple of two to create pairs of MPI ranks plus one rank for the diagonal elements
    if ((mpi_comm_size % 2) == 0) {
        throw std::runtime_error("Number of MPI ranks must be a multiple of 2 + 1");
    }
}


std::unique_ptr<transpose::TransposeData> transpose::DistributedExternalTransposeDataHandler::generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) {
    int width_in_blocks = settings.programSettings->matrixSize / settings.programSettings->blockSize;
    // The matrix without the diagonal blocks can be equally split between the configured MPI ranks and kernel replications
    bool two_matrix_sides_splitable = (((width_in_blocks * (width_in_blocks - 1) / 2)  % (mpi_comm_size * settings.programSettings->kernelReplications)) == 0);
    // The diagonal matrix blocks can be equally split between the configured ranks and replications
    bool diagonal_splitable = (width_in_blocks % (mpi_comm_size * settings.programSettings->kernelReplications)) == 0;
    // Check if data handler can be used with the given configuration
    if (!two_matrix_sides_splitable || !diagonal_splitable) {
        // The matrix is not equally divisible by the number of MPI ranks. Abort here.
        throw std::runtime_error("Matrix not equally divisible by number of MPI ranks. Choose a different data handler or change the MPI communication size to be a divisor of " + std::to_string(width_in_blocks * width_in_blocks));
    }
    int blocks_per_rank = width_in_blocks * width_in_blocks / (mpi_comm_size * settings.programSettings->kernelReplications);
    // Height of a matrix generated for a single memory bank on a single MPI rank
    int data_height_per_rank = blocks_per_rank * settings.programSettings->blockSize;
    
    // Allocate memory for a single device and all its memory banks
    auto d = std::unique_ptr<transpose::TransposeData>(new transpose::TransposeData(*settings.context, settings.programSettings->blockSize, blocks_per_rank,
                                                                                    settings.programSettings->kernelReplications));

    // Fill the allocated memory with pseudo random values
    std::mt19937 gen(7);
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    for (int r =0; r < settings.programSettings->kernelReplications; r++) {
        for (int i = 0; i < data_height_per_rank; i++) {
            for (int j = 0; j < settings.programSettings->blockSize; j++) {
                d->A[r][i * settings.programSettings->blockSize + j] = dis(gen);
                d->B[r][i * settings.programSettings->blockSize + j] = dis(gen);
                d->result[r][i * settings.programSettings->blockSize + j] = 0.0;
            }
        }
    }
    
    return d;
}

void transpose::DistributedExternalTransposeDataHandler::exchangeData(transpose::TransposeData& data) {
    // Calculate the rank of the paired FPGA
    int pair_rank = (mpi_comm_rank < mpi_comm_size / 2) ? (mpi_comm_size / 2 + mpi_comm_rank) : (mpi_comm_rank - mpi_comm_size / 2);

    // To re-calculate the matrix transposition locally on this host, we need to 
    // exchange matrix A for every kernel replication
    // The order of the matrix blocks does not change during the exchange, because they are distributed diagonally 
    // and will be handled in the order below:
    //
    // . . 1 3
    // . . . 2
    // 1 . . .
    // 3 2 . .
    for (HOST_DATA_TYPE* d : data.A) {
        MPI_Status status;
        MPI_Sendrecv_replace(d, data.blockSize * data.blockSize * data.numBlocks, MPI_FLOAT, pair_rank, 0, pair_rank, 0, MPI_COMM_WORLD, &status);
    }
}

#endif
