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
#include "transpose_data.hpp"

/**
 * @brief String that identifies the transpose::DistributedExternalTransposeDataHandler 
 * 
 */
#define TRANSPOSE_HANDLERS_DIST_EXT "distext" 

/**
 * @brief Contains all classes and methods needed by the Transpose benchmark
 * 
 */
namespace transpose {

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

    /**
     * @brief Construct a new Transpose Data Handler object and initialize the MPI rank and MPI size variables if MPI is used
     * 
     */
    TransposeDataHandler(int mpi_comm_rank, int mpi_comm_size) : mpi_comm_rank(mpi_comm_rank), mpi_comm_size(mpi_comm_size) {}

};

#ifdef _USE_MPI_

/**
 * @brief Transposes the data over external channels, so every part of a pair is located on a different FPGA.
 *         Data will be distributed to the ranks such that only a fixed pair of ranks will communicate to exchange
 *         the missing data. e.g. for N ranks, the pairs will be (0, N/2), (1, N/2 + 1), ...
 * 
 */
class DistributedExternalTransposeDataHandler : public transpose::TransposeDataHandler {

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
    generateData(hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& settings) override;

    /**
     * @brief Exchange the data blocks for verification
     * 
     * @param data The data that was generated locally and will be exchanged with other MPI ranks 
     *              Exchanged data will be stored in the same object.
     */
    void
    exchangeData(TransposeData& data) override;

    DistributedExternalTransposeDataHandler(int mpi_rank, int mpi_size);

};

#endif

/**
 * @brief Generate a data handler object
 * 
 * @tparam T The class of the data handler object
 * @return std::unique_ptr<transpose::TransposeDataHandler> a unique poiinter to the generated data handler object
 */
template<class T> 
std::unique_ptr<transpose::TransposeDataHandler> 
generateDataHandler(int rank, int size) {
    return std::unique_ptr<T>(new T(rank, size));
}

/**
 * @brief A map that contains the mapping from plain strings to the data handler object that should be used in the program
 * 
 */
extern std::map<std::string, std::unique_ptr<transpose::TransposeDataHandler> (*)(int rank, int size)> dataHandlerIdentifierMap;


}

#endif
