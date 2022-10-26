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
#ifndef SRC_HOST_ACCL_STREAM_PQ_SENDRECV_EXECUTION_H_
#define SRC_HOST_ACCL_STREAM_PQ_SENDRECV_EXECUTION_H_

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

void transpose_write_sendrecv(const DEVICE_DATA_TYPE* B,
                    DEVICE_DATA_TYPE* C,
                const int* target_list,
                int pq_row, int pq_col, 
                int pq_width, int pq_height,
                int gcd, int least_common_multiple,
                int height_per_rank,
                int width_per_rank,
                STREAM<stream_word> &cclo2krnl);
  
void transpose_read_sendrecv(const DEVICE_DATA_TYPE* A,
                const int* target_list,
                int pq_row, int pq_col, 
                int pq_width, int pq_height,
                int gcd, int least_common_multiple,
                int height_per_rank,
                int width_per_rank,
                STREAM<stream_word> &krnl2cclo);

namespace transpose {
namespace fpga_execution {
namespace accl_stream_sendrecv_pq {

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
static std::unique_ptr<transpose::TransposeExecutionTimings> calculate(
    const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings,
                                       xrt::device, bool, xrt::uuid> &config,
    transpose::TransposeData<bool> &data,
    transpose::data_handler::DistributedPQTransposeDataHandler<
        xrt::device, bool, xrt::uuid> &handler) {
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
  std::vector<std::unique_ptr<ACCL::Buffer<int>>> bufferListTargets;
  std::vector<std::unique_ptr<ACCL::Buffer<DEVICE_DATA_TYPE>>> bufferListCopy;
  std::vector<xrt::kernel> transposeReadKernelList;
  std::vector<xrt::kernel> transposeWriteKernelList;
  std::vector<size_t> blocksPerReplication;

  size_t local_matrix_width = handler.getWidthforRank();
  size_t local_matrix_height = handler.getHeightforRank();
  size_t local_matrix_width_bytes =
      local_matrix_width * data.blockSize * sizeof(HOST_DATA_TYPE);

  size_t total_offset = 0;
  size_t row_offset = 0;

  // Algorithm defines
  int pq_width = handler.getP();
  int pq_height = handler.getQ();

  int mpi_comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank);
  int mpi_comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  int pq_row = mpi_comm_rank / pq_width;
  int pq_col = mpi_comm_rank % pq_width;

  int gcd = std::__gcd(pq_height, pq_width);
  int least_common_multiple = pq_height * pq_width / gcd;

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

#ifndef NDEBUG
    std::cout << "Blocks per replication: " << blocks_per_replication << std::endl;
#endif

    row_offset = (row_offset + blocks_per_replication) % local_matrix_width;

    total_offset += (bufferOffsetList.back() + blocks_per_replication) /
                    local_matrix_width * local_matrix_width;

    // Pre-calculate target ranks in LCM block
    // The vector list variable can be interpreted as 2D matrix. Every entry
    // represents the target rank of the sub-block Since the LCM block will
    // repeat, we only need to store this small amount of data!
    auto target_list = config.accl->create_buffer<int>(least_common_multiple / pq_height *
                                least_common_multiple / pq_width, ACCL::dataType::int32);
    bufferListCopy.push_back(config.accl->create_buffer<DEVICE_DATA_TYPE>(buffer_size, ACCL::dataType::float32));
    for (int row = 0; row < least_common_multiple / pq_height; row++) {
      for (int col = 0; col < least_common_multiple / pq_width; col++) {
        int global_block_col = pq_col + col * pq_width;
        int global_block_row = pq_row + row * pq_height;
        int destination_rank = (global_block_col % pq_height) * pq_width +
                              (global_block_row % pq_width);
        target_list->buffer()[row * least_common_multiple / pq_width + col] =
            destination_rank;
      }
    }
    target_list->sync_to_device();
    bufferListTargets.push_back(std::move(target_list));

    if (!config.programSettings->useAcclEmulation) {
      // create the kernels
      xrt::kernel transposeReadKernel(
          *config.device, *config.program,
          ("transpose_read_sendrecv0:{transpose_read_sendrecv0_" + std::to_string(r + 1) + "}").c_str());
      xrt::kernel transposeWriteKernel(
          *config.device, *config.program,
          ("transpose_write_sendrecv0:{transpose_write_sendrecv0_" + std::to_string(r + 1) + "}").c_str());

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

#ifndef NDEBUG
    std::cout << "Start BFM" << std::endl;
#endif

    HLSLIB_DATAFLOW_INIT();
    hlslib::Stream<stream_word> cclo2krnl("cclo2krnl"), krnl2cclo("krnl2cclo");
    hlslib::Stream<command_word> cmd, sts;

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
        HLSLIB_DATAFLOW_FUNCTION(transpose_read_sendrecv,
            (config.programSettings->copyA ? data.A : data.A),
            bufferListTargets[r]->buffer(),
            pq_row, pq_col, pq_width, pq_height,
            gcd, least_common_multiple,
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize)),
                krnl2cclo);
        HLSLIB_DATAFLOW_FUNCTION(transpose_write_sendrecv,
            data.B, data.result,
            bufferListTargets[r]->buffer(),
            pq_row, pq_col, pq_width, pq_height,
            gcd, least_common_multiple,
            static_cast<cl_uint>(handler.getWidthforRank()),
            static_cast<cl_uint>(
                (bufferSizeList[r]) /
                (local_matrix_width * data.blockSize * data.blockSize)),
                cclo2krnl);
      }
    }
#ifndef NDEBUG
    std::cout << "Start ACCL send/recv" << std::endl;
#endif
    auto dbuffer = config.accl->create_buffer<DEVICE_DATA_TYPE>(1,ACCL::dataType::float32);
    int g = transpose::data_handler::mod(pq_row - pq_col, gcd);
    int p = transpose::data_handler::mod(pq_col + g, pq_width);
    int q = transpose::data_handler::mod(pq_row - g, pq_height);
    // Exchange A data via ACCL
    for (int k=0; k < 2; k++) {
      for (int j = 0; j < least_common_multiple/pq_width; j++) {
          for (int i = 0; i < least_common_multiple/pq_height; i++) {
              // Determine sender and receiver rank of current rank for current communication step
              int send_rank = transpose::data_handler::mod(p + i * gcd, pq_width) + transpose::data_handler::mod(q - j * gcd, pq_height) * pq_width;
              int recv_rank = transpose::data_handler::mod(p - i * gcd, pq_width) + transpose::data_handler::mod(q + j * gcd, pq_height) * pq_width;

              // Also count receiving buffer size because sending and receiving buffer size may differ in certain scenarios!
              int receiving_size = 0;
              int sending_size = 0;

              std::vector<int> send_rows;
              std::vector<int> send_cols;
              // Look up which blocks are affected by the current rank
              for (int row = 0; row  < least_common_multiple/pq_height; row++) {
                  for (int col = 0; col  < least_common_multiple/pq_width; col++) {
#ifndef NDEBUG
    std::cout << "Check" << row * least_common_multiple/pq_width + col << std::endl;
#endif
                      if (bufferListTargets[0]->buffer()[row * least_common_multiple/pq_width + col] == send_rank) {
                          send_rows.push_back(row);
                          send_cols.push_back(col);
                          sending_size += data.blockSize * data.blockSize;
                      }
                      if (bufferListTargets[0]->buffer()[row * least_common_multiple/pq_width + col] == recv_rank) {
                          receiving_size += data.blockSize * data.blockSize;
                      }
                  }
              }
              receiving_size *= (local_matrix_height)/(least_common_multiple/pq_height) * ((local_matrix_width)/(least_common_multiple/pq_width));
              sending_size *= (local_matrix_height)/(least_common_multiple/pq_height) * ((local_matrix_width)/(least_common_multiple/pq_width));

              // Do actual MPI communication
              if (k==0) {
                // First schedule all sends, then all receives. This works if communication rounds <= ACCL buffers.
                // Non-blocking communication would not offer many benefits, because the CCLO can only execute send OR recv
#ifndef NDEBUG
                  std::cout << "Send blocks " << sending_size / (data.blockSize * data.blockSize) << " to " << send_rank << std::endl << std::flush;
#endif
                  if (send_rank == mpi_comm_rank) {
                    //TODO copy from and to string not implemented in driver yet
                    // config.accl->copy_from_stream(*bufferListCopy[0], sending_size);
                  } else {
                    config.accl->send(ACCL::dataType::float32, sending_size, send_rank, 0);
                  }
              } else {
  #ifndef NDEBUG
                  std::cout << "Recv blocks " <<   receiving_size / (data.blockSize * data.blockSize) << " from " << recv_rank << std::endl << std::flush;
  #endif
                if (recv_rank == mpi_comm_rank) {
                  //TODO copy from and to string not implemented in driver yet
                  // config.accl->copy_to_stream(*bufferListCopy[0], receiving_size);
                } else {
                  config.accl->recv(ACCL::dataType::float32, receiving_size, recv_rank, 0);
                }
              }
          }
      }
    }

#ifndef NDEBUG
    std::cout << "Wait for kernels to complete" << std::endl;
#endif
    for (int r = 0; r < runs.size(); r++) {
      runs[r].wait();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    HLSLIB_DATAFLOW_FINALIZE();
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

  std::unique_ptr<transpose::TransposeExecutionTimings> result(
      new transpose::TransposeExecutionTimings{transferTimings,
                                               calculationTimings});

  return result;
}

} // namespace accl_pq
} // namespace fpga_execution
} // namespace transpose

#endif
