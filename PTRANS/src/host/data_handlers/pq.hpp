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

#ifndef SRC_HOST_TRANSPOSE_HANDLERS_PQ_HPP_
#define SRC_HOST_TRANSPOSE_HANDLERS_PQ_HPP_

/* C++ standard library headers */
#include <memory>

/* Project's headers */
#include "handler.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {
namespace data_handler {

class DistributedPQTransposeDataHandler : public TransposeDataHandler {

private:

    /**
     * @brief Width of the local matrix of the current rank in blocks
     * 
     */
    int width_per_rank;

    /**
     * @brief Height of the local matrix of the current rank in blocks
     * 
     */
    int height_per_rank;

    /**
     * @brief Row of the current rank in the PQ grid
     * 
     */
    int pq_row;

    /**
     * @brief Column of the current rank in the PQ grid
     * 
     */
    int pq_col;

    /**
     * @brief Width of the PQ grid (number of columns in PQ grid)
     * 
     */
    int pq_width;

    /**
     * @brief MPI derived data type for block-wise matrix transfer
     * 
     */
    MPI_Datatype data_block;

public:

    int getWidthforRank() {
        return width_per_rank;
    }

    int getHeightforRank() {
        return height_per_rank;
    }

    /**
     * @brief Generate data for transposition based on the implemented distribution scheme
     * 
     * @param settings The execution settings that contain information about the data size
     * @return std::unique_ptr<TransposeData> The generated data
     */
    std::unique_ptr<TransposeData>
    generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) override {
        int width_in_blocks = settings.programSettings->matrixSize / settings.programSettings->blockSize;

        // A data block is strided!
        MPI_Type_contiguous(settings.programSettings->blockSize * settings.programSettings->blockSize, MPI_FLOAT, &data_block);
        MPI_Type_commit(&data_block);

        width_per_rank = width_in_blocks / pq_width;
        pq_row = mpi_comm_rank / pq_width;
        pq_col = mpi_comm_rank % pq_width;

        // If the torus width is not a divisor of the matrix size,
        // distribute remaining blocks to the ranks
        int remainder_per_rank = width_in_blocks % pq_width;
        height_per_rank = (pq_row < remainder_per_rank) ? width_per_rank + 1 : width_per_rank;
        width_per_rank += (pq_col < remainder_per_rank) ? 1 : 0;

        int blocks_per_rank = height_per_rank * width_per_rank;
        
        // Allocate memory for a single device and all its memory banks
        auto d = std::unique_ptr<transpose::TransposeData>(new transpose::TransposeData(*settings.context, settings.programSettings->blockSize, blocks_per_rank));

        // Fill the allocated memory with pseudo random values
        std::mt19937 gen(mpi_comm_rank);
        std::uniform_real_distribution<> dis(-100.0, 100.0);
        for (size_t i = 0; i < blocks_per_rank * settings.programSettings->blockSize; i++) {
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
    exchangeData(TransposeData& data) override {

        if (pq_col != pq_row) {

            int pair_rank = pq_width * pq_col + pq_row;

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

    }

    void 
    reference_transpose(TransposeData& data) {
        for (size_t j = 0; j < height_per_rank * data.blockSize; j++) {
            for (size_t i = 0; i < width_per_rank * data.blockSize; i++) {
                data.A[i * height_per_rank * data.blockSize + j] -= (data.result[j * width_per_rank * data.blockSize + i] - data.B[j * width_per_rank * data.blockSize + i]);
            }
        }
    }

    DistributedPQTransposeDataHandler(int mpi_rank, int mpi_size) : TransposeDataHandler(mpi_rank, mpi_size) {
        int sqrt_size = std::sqrt(mpi_size);
        if (sqrt_size * sqrt_size != mpi_size) {
            throw std::runtime_error("Number of MPI ranks must have an integer as square root since P = Q has to hold!");
        }
        pq_width = std::sqrt(mpi_size);
    }

};


}
}

#endif
