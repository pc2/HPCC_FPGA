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

#ifndef SRC_HOST_TRANSPOSE_HANDLERS_HPP_
#define SRC_HOST_TRANSPOSE_HANDLERS_HPP_

/* C++ standard library headers */
#include <memory>

/* Project's headers */
#include "../transpose_data.hpp"

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {
namespace data_handler {
/**
 * @brief The parallel matrix transposition is designed to support different kinds of data distribution.
 *          This abstract class provides the necessary methods that need to be implemented for every data distribution scheme.
 *          In general, data will be generated locally on the device and blocks will be exchanged between the MPI ranks according to
 *          the used  data distribution scheme to allow local verification. Only the calculated error will be collected by rank 0 to
 *          calculate the overall validation error.
 * 
 */
class TransposeDataHandler {

protected:

    /**
     * @brief Rank in the MPI communication world
     * 
     */
    int mpi_comm_rank = 0;

    /**
     * @brief Total size of the MPI communication world
     * 
     */
    int mpi_comm_size = 1;

public:

    /**
     * @brief Generate data for transposition based on the implemented distribution scheme
     * 
     * @param settings The execution settings that contain information about the data size
     * @return std::unique_ptr<TransposeData> The generated data
     */
    virtual std::unique_ptr<TransposeData>
    generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) = 0;

    /**
     * @brief Exchange the data blocks for verification
     * 
     * @param data The data that was generated locally and will be exchanged with other MPI ranks. 
     *              Exchanged data will be stored in the same object.
     */
    virtual void
    exchangeData(TransposeData& data) = 0;

    virtual void
    reference_transpose(TransposeData& data) = 0;

    /**
     * @brief Construct a new Transpose Data Handler object and initialize the MPI rank and MPI size variables if MPI is used
     * 
     */
    TransposeDataHandler(int mpi_comm_rank, int mpi_comm_size) : mpi_comm_rank(mpi_comm_rank), mpi_comm_size(mpi_comm_size) {}

};

}
}

#endif
