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
     * @brief Number of diagonal ranks that will sent the blcoks to themselves
     * 
     */
    int num_diagonal_ranks;

public:

    /**
     * @brief Generate data for transposition based on the implemented distribution scheme
     * 
     * @param settings The execution settings that contain information about the data size
     * @return std::unique_ptr<TransposeData> The generated data
     */
    std::unique_ptr<TransposeData>
    generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) override {
        int width_in_blocks = settings.programSettings->matrixSize / settings.programSettings->blockSize;
        
        // // Allocate memory for a single device and all its memory banks
        // auto d = std::unique_ptr<transpose::TransposeData>(new transpose::TransposeData(*settings.context, settings.programSettings->blockSize, blocks_per_rank));

        // // Fill the allocated memory with pseudo random values
        // std::mt19937 gen(mpi_comm_rank);
        // std::uniform_real_distribution<> dis(-100.0, 100.0);
        // for (size_t i = 0; i < data_height_per_rank; i++) {
        //     for (size_t j = 0; j < settings.programSettings->blockSize; j++) {
        //         d->A[i * settings.programSettings->blockSize + j] = dis(gen);
        //         d->B[i * settings.programSettings->blockSize + j] = dis(gen);
        //         d->result[i * settings.programSettings->blockSize + j] = 0.0;
        //     }
        // }
        
        // return d;
    }

    /**
     * @brief Exchange the data blocks for verification
     * 
     * @param data The data that was generated locally and will be exchanged with other MPI ranks 
     *              Exchanged data will be stored in the same object.
     */
    void
    exchangeData(TransposeData& data) override {

    }

    DistributedPQTransposeDataHandler(int mpi_rank, int mpi_size) : TransposeDataHandler(mpi_rank, mpi_size) {

    }

};


}
}

#endif
