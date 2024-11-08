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
#ifndef SRC_HOST_ACCL_STREAM_PQ_EXECUTION_H_
#define SRC_HOST_ACCL_STREAM_PQ_EXECUTION_H_

/* C++ standard library headers */
#include <chrono>
#include <memory>
#include <vector>

/* Project's headers */
#include "data_handlers/data_handler_types.h"
#include "data_handlers/pq.hpp"
#include "transpose_data.hpp"
#include "cclo_bfm.h"
#include "Simulation.h"
#include "accl.hpp"

extern void transpose_write0(const DEVICE_DATA_TYPE *B,
                                 DEVICE_DATA_TYPE *A_out,
            const unsigned int offset_b,
            const unsigned int number_of_blocks,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks,
            hlslib::Stream<stream_word> &cclo2krnl);
  
extern void transpose_read0( const DEVICE_DATA_TYPE *A,
            const unsigned int offset_a,
            const unsigned int number_of_blocks,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks,
            hlslib::Stream<stream_word> &krnl2cclo);

namespace transpose {
namespace fpga_execution {
namespace accl_stream_pq {

/**
 * @brief Transpose and add the matrices using the OpenCL kernel using a PQ
 * distribution and PCIe+MPI over the host for communication
 *
 * @param config The progrma configuration
 * @param data data object that contains all required data for the execution on
 * the FPGA
 * @param handler data handler instance that should be used to exchange data
 * between hosts
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured
 * execution times
 */
static std::map<std::string, std::vector<double>>  calculate(
    const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings,
                                       xrt::device, fpga_setup::ACCLContext, xrt::uuid> &config,
    transpose::TransposeData<fpga_setup::ACCLContext> &data,
    transpose::data_handler::DistributedPQTransposeDataHandler<
        xrt::device, fpga_setup::ACCLContext, xrt::uuid> &handler) {
  int err;

  if (config.programSettings->dataHandlerIdentifier !=
      transpose::data_handler::DataHandlerType::pq) {
    throw std::runtime_error(
        "Used data handler not supported by execution handler!");
  }
#ifdef USE_SVM
  throw new std::runtime_error("SVM not supported in the host implementation "
                               "of this communication method");
#endif
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
  throw new std::runtime_error(
      "Using the Write Rect method is not supported in this host "
      "implementation of this communication method");
#endif

  std::vector<size_t> bufferSizeList;
  std::vector<size_t> bufferStartList;
  std::vector<size_t> bufferOffsetList;
  std::vector<xrt::bo> bufferListA;
  std::vector<xrt::bo> bufferListB;
  std::vector<xrt::bo> bufferListA_out;
  std::vector<xrt::kernel> transposeReadKernelList;
  std::vector<xrt::kernel> transposeWriteKernelList;
  std::vector<size_t> blocksPerReplication;

  size_t local_matrix_width = handler.getWidthforRank();
  size_t local_matrix_height = handler.getHeightforRank();
  size_t local_matrix_width_bytes =
      local_matrix_width * data.blockSize * sizeof(HOST_DATA_TYPE);

  size_t total_offset = 0;
  size_t row_offset = 0;
#ifndef NDEBUG
  std::cout << "Start kernel creation" << std::endl;
#endif
  // Setup the kernels depending on the number of kernel replications
  for (int r = 0; r < config.programSettings->kernelReplications; r++) {

    // Calculate how many blocks the current kernel replication will need to
    // process.
    size_t blocks_per_replication =
        (local_matrix_height * local_matrix_width /
         config.programSettings->kernelReplications);
    size_t blocks_remainder = (local_matrix_height * local_matrix_width) %
                              config.programSettings->kernelReplications;
    if (blocks_remainder > r) {
      // Catch the case, that the number of blocks is not divisible by the
      // number of kernel replications
      blocks_per_replication += 1;
    }
    if (blocks_per_replication < 1) {
      continue;
    }
    blocksPerReplication.push_back(blocks_per_replication);
    size_t buffer_size = (blocks_per_replication + local_matrix_width - 1) /
                         local_matrix_width * local_matrix_width *
                         data.blockSize * data.blockSize;
    bufferSizeList.push_back(buffer_size);
    bufferStartList.push_back(total_offset);
    bufferOffsetList.push_back(row_offset);

    row_offset = (row_offset + blocks_per_replication) % local_matrix_width;

    total_offset += (bufferOffsetList.back() + blocks_per_replication) /
                    local_matrix_width * local_matrix_width;

    if (!config.programSettings->useAcclEmulation) {
      // create the kernels
      xrt::kernel transposeReadKernel(
          *config.device, *config.program,
          ("transpose_read0:{transpose_read0_" + std::to_string(r + 1) + "}").c_str());
      xrt::kernel transposeWriteKernel(
          *config.device, *config.program,
          ("transpose_write0:{transpose_write0_" + std::to_string(r + 1) + "}").c_str());

      if (r == 0 || config.programSettings->copyA) {
        xrt::bo bufferA(*config.device, data.A,
                      data.numBlocks * data.blockSize * data.blockSize *
                          sizeof(HOST_DATA_TYPE),
                      transposeReadKernel.group_id(0));
        bufferListA.push_back(bufferA);
      }
      xrt::bo bufferB(
          *config.device,
          &data.B[bufferStartList[r] * data.blockSize * data.blockSize],
          buffer_size * sizeof(HOST_DATA_TYPE), transposeWriteKernel.group_id(0));
      xrt::bo bufferA_out(*config.device, buffer_size * sizeof(HOST_DATA_TYPE),
                          transposeWriteKernel.group_id(1));

      bufferListB.push_back(bufferB);
      bufferListA_out.push_back(bufferA_out);
      transposeReadKernelList.push_back(transposeReadKernel);
      transposeWriteKernelList.push_back(transposeWriteKernel);
    }
  }

  std::vector<double> transferTimings;
  std::vector<double> calculationTimings;

  for (int repetition = 0; repetition < config.programSettings->numRepetitions;
       repetition++) {

#ifndef NDEBUG
    std::cout << "Start data transfer" << std::endl;
#endif
    auto startTransfer = std::chrono::high_resolution_clock::now();

    if (!config.programSettings->useAcclEmulation) {
      for (int r = 0; r < transposeReadKernelList.size(); r++) {
        if (r == 0 || config.programSettings->copyA) {
          bufferListA[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        bufferListB[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    }
    auto endTransfer = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> transferTime =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            endTransfer - startTransfer);

    MPI_Barrier(MPI_COMM_WORLD);

    HLSLIB_DATAFLOW_INIT();
    hlslib::Stream<stream_word> cclo2krnl("cclo2krnl"), krnl2cclo("krnl2cclo");
    hlslib::Stream<command_word> cmd, sts;

    int pq_width = handler.getP();

    int mpi_comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank);
    int mpi_comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
    int pq_row = mpi_comm_rank / pq_width;
    int pq_col = mpi_comm_rank % pq_width;

    int pair_rank = pq_width * pq_col + pq_row;
    std::vector<unsigned int> dest = {0, 9};
    std::unique_ptr<CCLO_BFM> cclo;
    if (config.programSettings->useAcclEmulation) {
#ifndef NDEBUG
      std::cout << "Start BFM" << std::endl;
#endif
      cclo = std::make_unique<CCLO_BFM>(6000, mpi_comm_rank, mpi_comm_size, dest, cmd, sts, cclo2krnl, krnl2cclo);
      cclo->run();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto startCalculation = std::chrono::high_resolution_clock::now();

#ifndef NDEBUG
    std::cout << "Start kernel execution" << std::endl;
#endif
    std::vector<xrt::run> runs;
    auto startKernelCalculation = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < config.programSettings->kernelReplications; r++) {
      if (!config.programSettings->useAcclEmulation) {
        runs.push_back(transposeReadKernelList[r](
            (config.programSettings->copyA ? bufferListA[r] : bufferListA[0]),
            static_cast<cl_uint>(bufferOffsetList[r]),
            static_cast<cl_uint>(blocksPerReplication[r]),
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize))));
        runs.push_back(transposeWriteKernelList[r](
            bufferListB[r], bufferListA_out[r],
            static_cast<cl_uint>(bufferOffsetList[r]),
            static_cast<cl_uint>(blocksPerReplication[r]),
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize))));
      } else {
        HLSLIB_DATAFLOW_FUNCTION(transpose_read0,
            (config.programSettings->copyA ? data.A : data.A),
            static_cast<cl_uint>(bufferOffsetList[r]),
            static_cast<cl_uint>(blocksPerReplication[r]),
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize)),
                krnl2cclo);
        HLSLIB_DATAFLOW_FUNCTION(transpose_write0,
            data.B, data.result,
            static_cast<cl_uint>(bufferOffsetList[r]),
            static_cast<cl_uint>(blocksPerReplication[r]),
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize)),
                cclo2krnl);
      }
    }
    // Exchange A data via ACCL
    config.context->accl->stream_put(ACCL::dataType::float32, data.blockSize * data.blockSize * data.numBlocks,
                   pair_rank, 0);
#ifndef NDEBUG
    std::cout << "Wait for kernels to complete" << std::endl;
#endif
    for (int r = 0; r < runs.size(); r++) {
      runs[r].wait();
    }
    HLSLIB_DATAFLOW_FINALIZE();
    MPI_Barrier(MPI_COMM_WORLD);
    if (config.programSettings->useAcclEmulation) {
      cclo->stop();
    }
    auto endCalculation = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    std::cout << "Rank " << mpi_rank << ": "
              << "Done i=" << repetition << std::endl;
    std::cout << "Kernel execution time: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     endCalculation - startKernelCalculation)
                     .count()
              << "s ("
              << ((config.programSettings->matrixSize *
                   config.programSettings->matrixSize * sizeof(HOST_DATA_TYPE) *
                   3) /
                  std::chrono::duration_cast<std::chrono::duration<double>>(
                      endCalculation - startKernelCalculation)
                      .count() *
                  1.0e-9)
              << " GB/s)" << std::endl;
#endif

    std::chrono::duration<double> calculationTime =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            endCalculation - startCalculation);
    calculationTimings.push_back(calculationTime.count());

    std::vector<HOST_DATA_TYPE> tmp_write_buffer(
        local_matrix_height * local_matrix_width * data.blockSize *
        data.blockSize);

    startTransfer = std::chrono::high_resolution_clock::now();
    if (!config.programSettings->useAcclEmulation) {
      for (int r = 0; r < transposeReadKernelList.size(); r++) {
        // Copy possibly incomplete first block row
        if (bufferOffsetList[r] != 0) {
          bufferListA_out[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          bufferListA_out[r].read(tmp_write_buffer.data());
          for (int row = 0; row < data.blockSize; row++) {
            for (int col = bufferOffsetList[r] * data.blockSize;
                col < local_matrix_width * data.blockSize; col++) {
              data.result[bufferStartList[r] * data.blockSize * data.blockSize +
                          row * local_matrix_width * data.blockSize + col] =
                  tmp_write_buffer[row * local_matrix_width * data.blockSize +
                                  col];
            }
          }
          // Copy remaining buffer
          std::copy(tmp_write_buffer.begin() +
                        local_matrix_width * data.blockSize * data.blockSize,
                    tmp_write_buffer.begin() + bufferSizeList[r],
                    &data.result[(bufferStartList[r] + local_matrix_width) *
                                data.blockSize * data.blockSize]);
        } else {
          bufferListA_out[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          bufferListA_out[r].read(
              &data.result[bufferStartList[r] * data.blockSize * data.blockSize]);
        }
      }
    }
    endTransfer = std::chrono::high_resolution_clock::now();

    transferTime += std::chrono::duration_cast<std::chrono::duration<double>>(
        endTransfer - startTransfer);
    transferTimings.push_back(transferTime.count());
  }

  std::map<std::string, std::vector<double>> timings;
  timings["transfer"] = transferTimings;
  timings["calculation"] = calculationTimings;
  return timings;
}

} // namespace accl_pq
} // namespace fpga_execution
} // namespace transpose

#endif
