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
#ifndef SRC_HOST_ACCL_PQ_EXECUTION_H_
#define SRC_HOST_ACCL_PQ_EXECUTION_H_

/* C++ standard library headers */
#include <chrono>
#include <memory>
#include <vector>

/* Project's headers */
#include "buffer.hpp"
#include "cclo.hpp"
#include "constants.hpp"
#include "data_handlers/data_handler_types.h"
#include "data_handlers/pq.hpp"
#include "fpgabuffer.hpp"
#include "transpose_data.hpp"

namespace transpose {
namespace fpga_execution {
namespace accl_pq {

void accl_exchangeData(
    ACCL::ACCL &accl,
    transpose::data_handler::DistributedPQTransposeDataHandler<
        xrt::device, bool, xrt::uuid> &handler,
    transpose::TransposeData<bool> &data, std::vector<xrt::bo> &bufferAXrt,
    int global_width) {

  int pq_width = handler.getP();
  int pq_height = handler.getQ();
  int width_per_rank = handler.getWidthforRank();
  int height_per_rank = handler.getHeightforRank();

  int mpi_comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank);
  int pq_row = mpi_comm_rank / pq_width;
  int pq_col = mpi_comm_rank % pq_width;

  std::vector<std::unique_ptr<ACCL::BaseBuffer>> acclBuffersA;
  for (auto &bo : bufferAXrt) {
    acclBuffersA.push_back(accl.create_buffer<HOST_DATA_TYPE>(
        bo, data.blockSize * data.blockSize * data.numBlocks,
        ACCL::dataType::float32));
  }

  if (pq_width == pq_height) {
    if (pq_col != pq_row) {

      int pair_rank = pq_width * pq_col + pq_row;

      // To re-calculate the matrix transposition locally on this host, we need
      // to exchange matrix A for every kernel replication The order of the
      // matrix blocks does not change during the exchange, because they are
      // distributed diagonally and will be handled in the order below:
      //
      // . . 1 3
      // . . . 2
      // 1 . . .
      // 3 2 . .
      // auto AcclBufferA_recv = accl.create_buffer<HOST_DATA_TYPE>(
      //     data.blockSize * data.blockSize * data.numBlocks,
      //     ACCL::dataType::float32);
      // AcclBufferA_recv->sync_to_device();
      // Send and receive matrix A using ACCL directly on FPGA
      accl.send(0, *acclBuffersA[0],
                data.blockSize * data.blockSize * data.numBlocks, pair_rank, 0,
                true, ACCL::streamFlags::NO_STREAM);
      accl.recv(0, *acclBuffersA[0],
                data.blockSize * data.blockSize * data.numBlocks, pair_rank, 0,
                true, ACCL::streamFlags::NO_STREAM);
    }
  } else {
    // Taken from "Parallel matrix transpose algorithms on distributed memory
    // concurrent computers" by J. Choi, J. J. Dongarra, D. W. Walker and
    // translated to C++ This will do a diagonal exchange of matrix blocks.

    // Determine LCM using GCD from standard library using the C++14 call
    // In C++17 this changes to std::gcd in numeric, also std::lcm is directly
    // available in numeric
    int gcd = std::__gcd(pq_height, pq_width);
    int least_common_multiple = pq_height * pq_width / gcd;

    // If the global matrix size is not a multiple of the LCM block size, the
    // numbers of send and received blocks may be wrongly calculated. Throw
    // exception to prevent this and make aware of this issue!
    if (global_width % least_common_multiple > 0) {
      throw std::runtime_error(
          "Implementation does not support matrix sizes that are not multiple "
          "of LCM blocks! Results may be wrong!");
    }

    // MPI requests for non-blocking communication
    // First half of vector is for Isend, second half for Irecv!
    std::vector<ACCL::CCLO *> accl_requests(2 * gcd);

    // Begin algorithm from Figure 14 for general case
    int g = transpose::data_handler::mod(pq_row - pq_col, gcd);
    int p = transpose::data_handler::mod(pq_col + g, pq_width);
    int q = transpose::data_handler::mod(pq_row - g, pq_height);

    // Pre-calculate target ranks in LCM block
    // The vector list variable can be interpreted as 2D matrix. Every entry
    // represents the target rank of the sub-block Since the LCM block will
    // repeat, we only need to store this small amount of data!
    std::vector<int> target_list(least_common_multiple / pq_height *
                                 least_common_multiple / pq_width);
    for (int row = 0; row < least_common_multiple / pq_height; row++) {
      for (int col = 0; col < least_common_multiple / pq_width; col++) {
        int global_block_col = pq_col + col * pq_width;
        int global_block_row = pq_row + row * pq_height;
        int destination_rank = (global_block_col % pq_height) * pq_width +
                               (global_block_row % pq_width);
        target_list[row * least_common_multiple / pq_width + col] =
            destination_rank;
      }
    }

    // Create some ACCL buffers to send and receive from other FPGAs
    // They can reside completely on FPGA
    std::vector<std::unique_ptr<ACCL::BaseBuffer>> send_buffers;
    std::vector<std::unique_ptr<ACCL::BaseBuffer>> recv_buffers;
    for (int i = 0; i < gcd; i++) {
      // TODO Is there a way to initialize buffer only in FPGA memory with ACCL?
      send_buffers.push_back(accl.create_buffer<HOST_DATA_TYPE>(
          data.blockSize * data.blockSize * data.numBlocks,
          ACCL::dataType::float32));
      recv_buffers.push_back(accl.create_buffer<HOST_DATA_TYPE>(
          data.blockSize * data.blockSize * data.numBlocks,
          ACCL::dataType::float32));
      send_buffers.back()->sync_to_device();
      recv_buffers.back()->sync_to_device();
    }
    int current_parallel_execution = 0;
    for (int j = 0; j < least_common_multiple / pq_width; j++) {
      for (int i = 0; i < least_common_multiple / pq_height; i++) {
        // Determine sender and receiver rank of current rank for current
        // communication step
        int send_rank =
            transpose::data_handler::mod(p + i * gcd, pq_width) +
            transpose::data_handler::mod(q - j * gcd, pq_height) * pq_width;
        int recv_rank =
            transpose::data_handler::mod(p - i * gcd, pq_width) +
            transpose::data_handler::mod(q + j * gcd, pq_height) * pq_width;

        // Also count receiving buffer size because sending and receiving buffer
        // size may differ in certain scenarios!
        int receiving_size = 0;
        int sending_size = 0;

        std::vector<int> send_rows;
        std::vector<int> send_cols;
        // Look up which blocks are affected by the current rank
        for (int row = 0; row < least_common_multiple / pq_height; row++) {
          for (int col = 0; col < least_common_multiple / pq_width; col++) {
            if (target_list[row * least_common_multiple / pq_width + col] ==
                send_rank) {
              send_rows.push_back(row);
              send_cols.push_back(col);
              sending_size += data.blockSize * data.blockSize;
            }
            if (target_list[row * least_common_multiple / pq_width + col] ==
                recv_rank) {
              receiving_size += data.blockSize * data.blockSize;
            }
          }
        }
        receiving_size *=
            (height_per_rank) / (least_common_multiple / pq_height) *
            ((width_per_rank) / (least_common_multiple / pq_width));
        sending_size *= (height_per_rank) /
                        (least_common_multiple / pq_height) *
                        ((width_per_rank) / (least_common_multiple / pq_width));

#ifndef NDEBUG
        std::cout << "Copy data to send buffers" << std::endl;
#endif
        // Copy the required date for this communication step to the send
        // buffer!
        for (int t = 0; t < send_rows.size(); t++) {
          for (int lcm_row = 0;
               lcm_row <
               (height_per_rank) / (least_common_multiple / pq_height);
               lcm_row++) {
            for (int lcm_col = 0;
                 lcm_col <
                 (width_per_rank) / (least_common_multiple / pq_width);
                 lcm_col++) {
              size_t sending_buffer_offset =
                  lcm_row * data.blockSize * data.blockSize *
                      ((width_per_rank) / (least_common_multiple / pq_width)) +
                  lcm_col * data.blockSize * data.blockSize;
              size_t matrix_buffer_offset =
                  (send_cols[t] + lcm_col * least_common_multiple / pq_width) *
                      data.blockSize +
                  (send_rows[t] + lcm_row * least_common_multiple / pq_height) *
                      width_per_rank * data.blockSize * data.blockSize;
              for (int block_row = 0; block_row < data.blockSize; block_row++) {
                // TODO May be more efficient when done async!
                std::cout << "A("
                          << matrix_buffer_offset +
                                 block_row * width_per_rank * data.blockSize
                          << ","
                          << matrix_buffer_offset +
                                 block_row * width_per_rank * data.blockSize +
                                 data.blockSize
                          << ") send(" << sending_buffer_offset << ","
                          << sending_buffer_offset + data.blockSize << ")"
                          << std::endl;
                accl.copy(*acclBuffersA[0]->slice(
                              matrix_buffer_offset +
                                  block_row * width_per_rank * data.blockSize,
                              matrix_buffer_offset +
                                  block_row * width_per_rank * data.blockSize +
                                  data.blockSize),
                          *send_buffers[current_parallel_execution]->slice(
                              sending_buffer_offset,
                              sending_buffer_offset + data.blockSize),
                          data.blockSize, true, true);
                std::cout << "Copy done!" << std::endl;
              }
            }
          }
        }

        // Do actual MPI communication
#ifndef NDEBUG
        std::cout << "Rank " << mpi_comm_rank << ": blocks ("
                  << sending_size / (data.blockSize * data.blockSize) << ","
                  << receiving_size / (data.blockSize * data.blockSize)
                  << ") send " << send_rank << ", recv " << recv_rank
                  << std::endl
                  << std::flush;
#endif
        accl_requests[current_parallel_execution] = (accl.send(
            0, *send_buffers[current_parallel_execution], sending_size,
            send_rank, 0, true, ACCL::streamFlags::NO_STREAM, true));
        accl_requests[current_parallel_execution + gcd] = (accl.recv(
            0, *recv_buffers[current_parallel_execution], sending_size,
            send_rank, 0, true, ACCL::streamFlags::NO_STREAM, true));
        // Increase the counter for parallel executions
        current_parallel_execution = (current_parallel_execution + 1) % gcd;

        // Wait for MPI requests if GCD MPI calls are scheduled in parallel
        if ((current_parallel_execution) % gcd == 0) {

          for (auto &req : accl_requests) {

            MPI_Status status;
            int index;
#ifndef NDEBUG
            std::cout << "Wait for all requests to complete" << std::endl;
#endif
            // Wait for all send and recv events to complete
            // TODO do the CCLO pointers need to be freed?
            accl.nop(false, accl_requests);
            // For each message that was received in parallel
            if (index >= gcd) {
              std::vector<int> recv_rows;
              std::vector<int> recv_cols;
              // Look up which blocks are affected by the current rank
              for (int row = 0; row < least_common_multiple / pq_height;
                   row++) {
                for (int col = 0; col < least_common_multiple / pq_width;
                     col++) {
                  if (target_list[row * least_common_multiple / pq_width +
                                  col] == status.MPI_SOURCE) {
                    recv_rows.push_back(row);
                    recv_cols.push_back(col);
                  }
                }
              }
              // Copy received data to matrix A buffer
              for (int t = 0; t < recv_rows.size(); t++) {
                for (int lcm_row = 0;
                     lcm_row <
                     (height_per_rank) / (least_common_multiple / pq_height);
                     lcm_row++) {
                  for (int lcm_col = 0;
                       lcm_col <
                       (width_per_rank) / (least_common_multiple / pq_width);
                       lcm_col++) {
                    size_t receiving_buffer_offset =
                        lcm_row * data.blockSize * data.blockSize *
                            ((width_per_rank) /
                             (least_common_multiple / pq_width)) +
                        lcm_col * data.blockSize * data.blockSize;
                    size_t matrix_buffer_offset =
                        (recv_cols[t] +
                         lcm_col * least_common_multiple / pq_width) *
                            data.blockSize +
                        (recv_rows[t] +
                         lcm_row * least_common_multiple / pq_height) *
                            width_per_rank * data.blockSize * data.blockSize;
                    for (int block_row = 0; block_row < data.blockSize;
                         block_row++) {
                      // TODO May be more efficient when done async!
                      accl.copy(
                          *recv_buffers[current_parallel_execution]->slice(
                              receiving_buffer_offset,
                              receiving_buffer_offset + data.blockSize),
                          *acclBuffersA[0]->slice(
                              matrix_buffer_offset +
                                  block_row * width_per_rank * data.blockSize,
                              matrix_buffer_offset +
                                  block_row * width_per_rank * data.blockSize +
                                  data.blockSize),
                          data.blockSize, true, true);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Copy received matrix A to the buffers of other kernel replications that
  // may be placed on different memory banks
  for (int b = 1; b < acclBuffersA.size(); b++) {
    accl.copy(*acclBuffersA[0], *acclBuffersA[b],
              data.blockSize * data.blockSize * data.numBlocks, true, true);
  }
}

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
  std::vector<xrt::kernel> transposeKernelList;
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

    // create the kernels
    xrt::kernel transposeKernel(
        *config.device, *config.program,
        ("transpose0:{transpose0_" + std::to_string(r + 1) + "}").c_str());

    xrt::bo bufferA(*config.device, data.A,
                    data.numBlocks * data.blockSize * data.blockSize *
                        sizeof(HOST_DATA_TYPE),
                    transposeKernel.group_id(0));
    xrt::bo bufferB(
        *config.device,
        &data.B[bufferStartList[r] * data.blockSize * data.blockSize],
        buffer_size * sizeof(HOST_DATA_TYPE), transposeKernel.group_id(1));
    xrt::bo bufferA_out(*config.device, buffer_size * sizeof(HOST_DATA_TYPE),
                        transposeKernel.group_id(2));

    bufferListA.push_back(bufferA);
    bufferListB.push_back(bufferB);
    bufferListA_out.push_back(bufferA_out);
    transposeKernelList.push_back(transposeKernel);
  }

  std::vector<double> transferTimings;
  std::vector<double> calculationTimings;

  for (int repetition = 0; repetition < config.programSettings->numRepetitions;
       repetition++) {

#ifndef NDEBUG
    std::cout << "Start data transfer" << std::endl;
#endif
    auto startTransfer = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < transposeKernelList.size(); r++) {
      bufferListA[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bufferListB[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto endTransfer = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> transferTime =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            endTransfer - startTransfer);

    MPI_Barrier(MPI_COMM_WORLD);

    auto startCalculation = std::chrono::high_resolution_clock::now();

    // Exchange A data via ACCL
    if (bufferListA.size() > 1) {
      std::cerr << "WARNING: Only the matrix A of the first kernel replication "
                   "will be exchanged "
                   "via ACCL!"
                << std::endl;
    }
#ifndef NDEBUG
    std::cout << "Start data exchange with ACCL" << std::endl;
#endif
    accl_exchangeData(*config.accl, handler, data, bufferListA,
                      config.programSettings->matrixSize / data.blockSize);
#ifndef NDEBUG
    std::cout << "End data exchange with ACCL" << std::endl;
#endif
    std::vector<xrt::run> runs;
    auto startKernelCalculation = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < transposeKernelList.size(); r++) {
      runs.push_back(transposeKernelList[r](
          bufferListA[r], bufferListB[r], bufferListA_out[r],
          static_cast<cl_uint>(bufferOffsetList[r]),
          static_cast<cl_uint>(bufferOffsetList[r]),
          static_cast<cl_uint>(blocksPerReplication[r]),
          static_cast<cl_uint>(handler.getWidthforRank()),
          static_cast<cl_uint>(
              (bufferSizeList[r]) /
              (local_matrix_width * data.blockSize * data.blockSize))));
    }
#ifndef NDEBUG
    std::cout << "Wait for kernels to complete" << std::endl;
#endif
    for (int r = 0; r < transposeKernelList.size(); r++) {
      runs[r].wait();
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

    for (int r = 0; r < transposeKernelList.size(); r++) {
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
