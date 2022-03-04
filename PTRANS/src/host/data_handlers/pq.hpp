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
#include <algorithm>

/* Project's headers */
#include "handler.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {
namespace data_handler {

template<typename T> 
static T mod(T number, T op) {
    T result = number % op;
    return (result < 0) ? op + result : result;
}

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
     * @brief Global width and height of the matrix
     * 
     */
    int global_width;

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
     * @brief Height of the PQ grid (number of columns in PQ grid)
     * 
     */
    int pq_height;

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
        global_width = width_in_blocks;

        width_per_rank = width_in_blocks / pq_width;
        height_per_rank = width_in_blocks / pq_height;
        pq_row = mpi_comm_rank / pq_width;
        pq_col = mpi_comm_rank % pq_width;

        // If the torus width is not a divisor of the matrix size,
        // distribute remaining blocks to the ranks
        height_per_rank += (pq_row < (width_in_blocks % pq_height)) ?  1 : 0;
        width_per_rank += (pq_col < (width_in_blocks % pq_width)) ? 1 : 0;

        // A data block is strided and stride depends on the local matrix size!
        MPI_Type_vector(settings.programSettings->blockSize,settings.programSettings->blockSize,(width_per_rank - 1)*settings.programSettings->blockSize, MPI_FLOAT, &data_block);
        MPI_Type_commit(&data_block);

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

        MPI_Status status;     

        if (pq_width == pq_height) {
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
        else {
            // Taken from "Parallel matrix transpose algorithms on distributed memory concurrent computers" by J. Choi, J. J. Dongarra, D. W. Walker
            // and translated to C++
            // This will do a diagonal exchange of matrix blocks.

            // Determine LCM using GCD from standard library using the C++14 call
            // In C++17 this changes to std::gcd in numeric, also std::lcm is directly available in numeric
            int gcd = std::__gcd(pq_height, pq_width);
            int least_common_multiple = pq_height * pq_width / gcd;

            // Allocate some memory for sending and receiving data
            std::vector<HOST_DATA_TYPE> send_buffer(data.blockSize * data.blockSize * data.numBlocks);
            std::vector<HOST_DATA_TYPE> recv_buffer(data.blockSize * data.blockSize * data.numBlocks);
            // std::vector<HOST_DATA_TYPE> recv_buffer(data.blockSize * data.blockSize * ((global_width + least_common_multiple - 1) / least_common_multiple));

            // Begin algorithm from Figure 14 for general case
            // for (int parallel = 0; parallel < gcd; parallel++) {
                int g = mod(pq_row - pq_col, gcd);
                int p = mod(pq_col + g, pq_width);
                int q = mod(pq_row - g, pq_height);

                for (int j = 0; j < least_common_multiple/pq_width; j++) {
                    for (int i = 0; i < least_common_multiple/pq_height; i++) {
                        // Determine sender and receiver rank of current rank for current communication step
                        int send_rank = mod(p + i * gcd, pq_width) + mod(q - j * gcd, pq_height) * pq_width;
                        int recv_rank = mod(p - i * gcd, pq_width) + mod(q + j * gcd, pq_height) * pq_width;

                        // Collect all blocks that need to be send to other rank
                        auto send_buffer_location = send_buffer.begin();
                        for (int row = 0; row  < height_per_rank; row++) {
                            for (int col = 0; col  < width_per_rank; col++) {
                                // check for each local block, if its destinatio is the current send_rank
                                int global_block_col = pq_col + col * pq_width;
                                int global_block_row = pq_row + row * pq_height;
                                int destination_rank = (global_block_col % pq_height) * pq_width + (global_block_row % pq_width);
                                if (destination_rank == send_rank) {
                                    // add block to send buffer, if its destination matches send_rank
                                    size_t matrix_buffer_offset = col * data.blockSize  + row * width_per_rank * data.blockSize * data.blockSize;
                                    for (int block_row = 0; block_row < data.blockSize; block_row++) {
                                        std::copy(data.A + matrix_buffer_offset + block_row * width_per_rank * data.blockSize, data.A + matrix_buffer_offset + block_row * width_per_rank * data.blockSize + data.blockSize, send_buffer_location);
                                        send_buffer_location += data.blockSize;
                                    }
                                }
                            }
                        }

                        // Do actual MPI communication
                        int sending_size = send_buffer_location - send_buffer.begin();
                        std::cout << "Rank " << mpi_comm_rank << ": blocks " << sending_size / (data.blockSize * data.blockSize) << " send " << send_rank << ", recv " << recv_rank << std::endl << std::flush;
                        MPI_Sendrecv(send_buffer.data(), sending_size, MPI_FLOAT, send_rank, 0, recv_buffer.data(), sending_size, MPI_FLOAT, recv_rank, 0, MPI_COMM_WORLD, &status);
                        MPI_Barrier(MPI_COMM_WORLD);

                        // Insert received data into result matrix
                        auto recv_buffer_location = recv_buffer.begin();
                        for (int row = 0; row  < height_per_rank; row++) {
                            for (int col = 0; col  < width_per_rank; col++) {
                                // check for each local block, if its souce is the current recv_rank
                                int global_block_col = pq_col + col * pq_width;
                                int global_block_row = pq_row + row * pq_height;
                                int destination_rank = (global_block_col % pq_height) * pq_width + (global_block_row % pq_width);
                                if (destination_rank == recv_rank) {
                                    // add block from receive buffer to exchange buffer
                                    size_t matrix_buffer_offset = col * data.blockSize  + row * width_per_rank * data.blockSize * data.blockSize;
                                    for (int block_row = 0; block_row < data.blockSize; block_row++) {
                                        std::copy(recv_buffer_location, recv_buffer_location + data.blockSize,data.exchange + matrix_buffer_offset + block_row * width_per_rank * data.blockSize);
                                        recv_buffer_location += data.blockSize;
                                    }
                                }
                            }
                        }
                    }
                }
            // }
 
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

/**
 * @brief Construct a new Distributed P Q Transpose Data Handler object
 * 
 * @param mpi_rank MPI rank of the FPGA
 * @param mpi_size Size of the communication world
 * @param p Width of the PQ grid the FPGAs are arranged in
 */
    DistributedPQTransposeDataHandler(int mpi_rank, int mpi_size, int p) : TransposeDataHandler(mpi_rank, mpi_size) {
        if (mpi_size % p != 0) {
            throw std::runtime_error("Number of MPI ranks must be multiple of P! P=" + std::to_string(p));
        }
        pq_width = p;
        pq_height = mpi_size / p;
    }

};


}
}

#endif
