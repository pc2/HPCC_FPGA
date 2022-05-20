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
#ifndef EXECUTION_TYPES_EXECUTION_ACCL_BUFFERS_HPP
#define EXECUTION_TYPES_EXECUTION_ACCL_BUFFERS_HPP

/* C++ standard library headers */
#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <thread>
#include <vector>

/* External library headers */
#ifdef _OPENMP
#include "omp.h"
#endif

#include "linpack_data.hpp"
#include "parameters.h"

namespace linpack {
namespace execution {
namespace accl_buffers {

/*
 Prepare kernels and execute benchmark

 @copydoc bm_execution::calculate()
*/
std::unique_ptr<linpack::LinpackExecutionTimings> calculate(
    const hpcc_base::ExecutionSettings<linpack::LinpackProgramSettings,
                                       xrt::device, bool, xrt::uuid> &config,
    linpack::LinpackData &data) {

  cl_int err;

  int num_omp_threads = 1;
#ifdef _OPENMP
  num_omp_threads = omp_get_num_threads();
#endif

  uint blocks_per_row = data.matrix_width / config.programSettings->blockSize;
  uint blocks_per_col = data.matrix_height / config.programSettings->blockSize;

  // Communicate with all ranks in the same row of the torus
  // Configure ACCL Communicators

  // Get group of global communicator
  std::vector<ACCL::rank_t> all_accl_ranks =
      config.accl->get_comm_group(ACCL::GLOBAL_COMM);

  std::vector<ACCL::rank_t> row_ranks;
  std::vector<ACCL::rank_t> col_ranks;

  // Create sub-groups for rows and columns
  for (int i = config.programSettings->torus_width *
               config.programSettings->torus_row;
       i < config.programSettings->torus_width *
               (config.programSettings->torus_row + 1);
       i++) {
    row_ranks.push_back(all_accl_ranks[i]);
  }
  for (int i = config.programSettings->torus_col; i < all_accl_ranks.size();
       i += config.programSettings->torus_width) {
    col_ranks.push_back(all_accl_ranks[i]);
  }

  // Create communicators from sub-groups
  ACCL::communicatorId row_comm = config.accl->create_communicator(
      row_ranks, config.programSettings->torus_col);
  ACCL::communicatorId col_comm = config.accl->create_communicator(
      col_ranks, config.programSettings->torus_row);

  // Create global memory buffers
  auto lu_tmp_kernel = xrt::kernel(*config.device, *config.program, "lu");
  xrt::bo Buffer_a(*config.device, data.A,
                   sizeof(HOST_DATA_TYPE) * data.matrix_height *
                       data.matrix_width,
                   lu_tmp_kernel.group_id(0));
  xrt::bo Buffer_b(*config.device, data.b,
                   sizeof(HOST_DATA_TYPE) * data.matrix_width,
                   lu_tmp_kernel.group_id(0));
  xrt::bo Buffer_pivot(*config.device, data.ipvt,
                       sizeof(cl_int) * data.matrix_height,
                       lu_tmp_kernel.group_id(0));

  // TODO: To make this code work with the ACCL simulator, we need to create
  // buffers using bos. This vector is used to store these bos during execution.
  // They will be accessed via the ACCL buffers are not required in the code
  // itself. Fixing the simulator code of ACCL to always create a bo would fix
  // this issue.
  std::vector<xrt::bo> tmp_bos;

  /* --- Setup MPI communication and required additional buffers --- */

  // Buffers only used to store data received over the network layer
  // The content will not be modified by the host
  tmp_bos.emplace_back(*config.device,
                       sizeof(HOST_DATA_TYPE) *
                           (config.programSettings->blockSize) *
                           (config.programSettings->blockSize),
                       lu_tmp_kernel.group_id(1));
  auto Buffer_lu1 = config.accl->create_buffer<HOST_DATA_TYPE>(
      tmp_bos.back(),
      (config.programSettings->blockSize) * (config.programSettings->blockSize),
      ACCL::dataType::float32);
  tmp_bos.emplace_back(*config.device,
                       sizeof(HOST_DATA_TYPE) *
                           (config.programSettings->blockSize) *
                           (config.programSettings->blockSize),
                       lu_tmp_kernel.group_id(2));
  auto Buffer_lu2 = config.accl->create_buffer<HOST_DATA_TYPE>(
      tmp_bos.back(),
      (config.programSettings->blockSize) * (config.programSettings->blockSize),
      ACCL::dataType::float32);
  Buffer_lu1->sync_to_device();
  Buffer_lu2->sync_to_device();

  std::vector<std::vector<std::unique_ptr<ACCL::BaseBuffer>>> Buffer_left_list;
  std::vector<std::vector<std::unique_ptr<ACCL::BaseBuffer>>> Buffer_top_list;

  // Create two sets of communication buffers to allow overlap of communication
  // and matrix multiplications
  for (int rep = 0; rep < 2; rep++) {
    Buffer_left_list.emplace_back();
    Buffer_top_list.emplace_back();
    for (int i = 0; i < blocks_per_row; i++) {
      tmp_bos.emplace_back(*config.device,
                           sizeof(HOST_DATA_TYPE) *
                               (config.programSettings->blockSize) *
                               (config.programSettings->blockSize),
                           lu_tmp_kernel.group_id(0));
      Buffer_top_list.back().push_back(
          config.accl->create_buffer<HOST_DATA_TYPE>(
              tmp_bos.back(),
              (config.programSettings->blockSize) *
                  (config.programSettings->blockSize),
              ACCL::dataType::float32));
      Buffer_top_list.back().back()->sync_to_device();
    }

    for (int i = 0; i < blocks_per_col; i++) {
      tmp_bos.emplace_back(*config.device,
                           sizeof(HOST_DATA_TYPE) *
                               (config.programSettings->blockSize) *
                               (config.programSettings->blockSize),
                           lu_tmp_kernel.group_id(2));
      Buffer_left_list.back().push_back(
          config.accl->create_buffer<HOST_DATA_TYPE>(
              tmp_bos.back(),
              (config.programSettings->blockSize) *
                  (config.programSettings->blockSize),
              ACCL::dataType::float32));
      Buffer_left_list.back().back()->sync_to_device();
    }
  }

  /* --- Execute actual benchmark kernels --- */

  double t;
  std::vector<double> gefaExecutionTimes;
  std::vector<double> geslExecutionTimes;
  std::vector<double> gefaWaitTimes;
  for (int i = 0; i < config.programSettings->numRepetitions; i++) {

    Buffer_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    Buffer_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Command queues
    // A new command queue is created for every iteration of the algorithm to
    // reduce the overhead of too large queues
    std::vector<xrt::run> inner_mms;
    std::thread flush_thread;

    std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2, twait1,
        twait2;
    std::chrono::duration<double> currentwaittime =
        std::chrono::duration<double>::zero();

    std::cout << "Torus " << config.programSettings->torus_row << ","
              << config.programSettings->torus_col << "Start! " << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = std::chrono::high_resolution_clock::now();

    int kernel_offset = 0;
#pragma omp parallel
    {

#pragma omp single
      uint current_replication = 0;

      // For every row of blocks create kernels and enqueue them
      for (int block_row = 0; block_row < config.programSettings->matrixSize /
                                              config.programSettings->blockSize;
           block_row++) {

        int local_block_row_remainder =
            (block_row % config.programSettings->torus_height);
        int local_block_row =
            (block_row / config.programSettings->torus_height);
        int local_block_col_remainder =
            (block_row % config.programSettings->torus_width);
        int local_block_col = (block_row / config.programSettings->torus_width);
        bool in_same_row_as_lu =
            local_block_row_remainder == config.programSettings->torus_row;
        bool in_same_col_as_lu =
            local_block_col_remainder == config.programSettings->torus_col;
        int start_row_index =
            local_block_row +
            ((local_block_row_remainder >= config.programSettings->torus_row)
                 ? 1
                 : 0);
        int start_col_index =
            local_block_col +
            ((local_block_col_remainder >= config.programSettings->torus_col)
                 ? 1
                 : 0);
        int num_left_blocks =
            (in_same_col_as_lu) ? blocks_per_col - start_row_index : 0;
        int num_top_blocks =
            (in_same_row_as_lu) ? blocks_per_row - start_col_index : 0;
        int num_inner_block_rows = (blocks_per_col - start_row_index);
        int num_inner_block_cols =
            (num_inner_block_rows > 0) ? (blocks_per_row - start_col_index) : 0;
        num_inner_block_rows =
            (num_inner_block_cols > 0) ? num_inner_block_rows : 0;
        bool is_calulating_lu_block = (in_same_col_as_lu && in_same_row_as_lu);

#ifndef NDEBUG
        std::cout << "Torus " << config.programSettings->torus_row << ","
                  << config.programSettings->torus_col
                  << " Start iteration     " << block_row << std::endl;
#endif

        uint total_inner_updates_first_row = num_inner_block_cols;
        uint updates_per_replication =
            total_inner_updates_first_row /
            config.programSettings->kernelReplications;
        uint total_inner_updates =
            (num_inner_block_cols - 1) * (num_inner_block_rows - 1);
        uint total_updates_per_replication =
            total_inner_updates / config.programSettings->kernelReplications;
        uint current_update = 0;

        std::vector<xrt::run> comm_kernel_runs;

#pragma omp single
        {

          if (is_calulating_lu_block) {
            // create the LU kernel
            auto lu_kernel = xrt::kernel(*config.device, *config.program, "lu");

#ifndef NDEBUG
            std::cout << "Torus " << config.programSettings->torus_row << ","
                      << config.programSettings->torus_col << " LU     "
                      << local_block_row << "," << local_block_col << std::endl;
#endif
            auto lu_run =
                lu_kernel(Buffer_a, *Buffer_lu1->bo(), *Buffer_lu2->bo(),
                          local_block_col, local_block_row, blocks_per_row);
            ert_cmd_state state = lu_run.wait();
            if (state != ERT_CMD_STATE_COMPLETED) {
              std::cerr << "Execution Lu failed: " << state << std::endl;
            }
          }

          // Exchange LU blocks on all ranks to prevent stalls in MPI broadcast
          // All tasks until now need to be executed so we can use the result of
          // the LU factorization and communicate it via MPI with the other
          // FPGAs

          // Broadcast LU block in column to update all left blocks
          config.accl->bcast(*Buffer_lu2,
                             config.programSettings->blockSize *
                                 config.programSettings->blockSize,
                             local_block_row_remainder, col_comm, true, true);
          // Broadcast LU block in row to update all top blocks
          config.accl->bcast(*Buffer_lu1,
                             config.programSettings->blockSize *
                                 config.programSettings->blockSize,
                             local_block_col_remainder, row_comm, true, true);
        }
        if (num_top_blocks > 0) {

// Create top kernels
#pragma omp for
          for (int tops = start_col_index; tops < blocks_per_row; tops++) {
            xrt::kernel k(*config.device, *config.program, "top_update");
#ifndef NDEBUG
            std::cout << "Torus " << config.programSettings->torus_row << ","
                      << config.programSettings->torus_col << " Top    "
                      << local_block_row << "," << tops << std::endl;
#endif

            comm_kernel_runs.push_back(
                k(Buffer_a,
                  *Buffer_top_list[block_row % 2][tops - start_col_index]->bo(),
                  *Buffer_lu1->bo(), (tops == start_col_index), tops,
                  local_block_row, blocks_per_row));
          }
        }
        if (num_left_blocks > 0) {

// Create left kernels
#pragma omp for
          for (int tops = start_row_index; tops < blocks_per_col; tops++) {
            xrt::kernel k(*config.device, *config.program, "left_update");
#ifndef NDEBUG
            std::cout << "Torus " << config.programSettings->torus_row << ","
                      << config.programSettings->torus_col << " Left   " << tops
                      << "," << local_block_col << std::endl;
#endif
            comm_kernel_runs.push_back(k(
                Buffer_a,
                *Buffer_left_list[block_row % 2][tops - start_row_index]->bo(),
                *Buffer_lu2->bo(), (tops == start_row_index), local_block_col,
                tops, blocks_per_row));
          }
        }

#pragma omp single
        {
          // Wait until all top and left blocks are calculated
          for (auto &run : comm_kernel_runs) {
            run.wait();
          }

          // Send the left and top blocks to all other ranks so they can be used
          // to update all inner blocks
          for (int lbi = 0;
               lbi <
               std::max(static_cast<int>(blocks_per_col - local_block_col), 0);
               lbi++) {
            config.accl->bcast(*Buffer_left_list[block_row % 2][lbi],
                               config.programSettings->blockSize *
                                   config.programSettings->blockSize,
                               local_block_col_remainder, row_comm, true, true);
          }
          for (int tbi = 0;
               tbi <
               std::max(static_cast<int>(blocks_per_row - local_block_row), 0);
               tbi++) {
            config.accl->bcast(*Buffer_top_list[block_row % 2][tbi],
                               config.programSettings->blockSize *
                                   config.programSettings->blockSize,
                               local_block_row_remainder, col_comm, true, true);
          }
          // update all remaining inner blocks using only global memory
        }

        std::vector<xrt::run> outer_mms;

        // Wait for previous inner MMs to complete.
        // They may need to be reused by the next outer MM calls!
        for (auto &run : inner_mms) {
          run.wait();
        }

#pragma omp for
        for (int lbi = 1; lbi < num_inner_block_rows; lbi++) {

          // select the matrix multiplication kernel that should be used for
          // this block updated
          xrt::kernel k(*config.device, *config.program, "inner_update_mm0");

          int current_block_col = static_cast<cl_uint>(
              (data.matrix_width / config.programSettings->blockSize) -
              num_inner_block_cols);
          int current_block_row = static_cast<cl_uint>(
              (data.matrix_height / config.programSettings->blockSize) -
              num_inner_block_rows + lbi);

#ifndef NDEBUG
          std::cout << "Torus " << config.programSettings->torus_row << ","
                    << config.programSettings->torus_col << " MM col "
                    << current_block_row << "," << current_block_col
                    << std::endl;
#endif

          outer_mms.push_back(
              k(Buffer_a, *Buffer_left_list[block_row % 2][lbi]->bo(),
                *Buffer_top_list[block_row % 2][0]->bo(), current_block_col,
                current_block_row, blocks_per_row));
        }

#pragma omp for
        for (int tbi = 0; tbi < num_inner_block_cols; tbi++) {

          // select the matrix multiplication kernel that should be used for
          // this block updated
          xrt::kernel k(*config.device, *config.program, "inner_update_mm0");

          int current_block_col = static_cast<cl_uint>(
              (data.matrix_width / config.programSettings->blockSize) -
              num_inner_block_cols + tbi);
          int current_block_row = static_cast<cl_uint>(
              (data.matrix_height / config.programSettings->blockSize) -
              num_inner_block_rows);

#ifndef NDEBUG
          std::cout << "Torus " << config.programSettings->torus_row << ","
                    << config.programSettings->torus_col << " MM row "
                    << current_block_row << "," << current_block_col
                    << std::endl;
#endif

          outer_mms.push_back(
              k(Buffer_a, *Buffer_left_list[block_row % 2][0]->bo(),
                *Buffer_top_list[block_row % 2][tbi]->bo(), current_block_col,
                current_block_row, blocks_per_row));
        }

        // Clear inner MM runs vector for this iteration
        // All runs have completed before scheduling the outer MMs
        inner_mms.clear();

#pragma omp for collapse(2) schedule(static)
        for (int lbi = 1; lbi < num_inner_block_rows; lbi++) {
          for (int tbi = 1; tbi < num_inner_block_cols; tbi++) {
            // select the matrix multiplication kernel that should be used for
            // this block updated

            xrt::kernel k(*config.device, *config.program, "inner_update_mm0");

            int current_block_col = static_cast<cl_uint>(
                (data.matrix_width / config.programSettings->blockSize) -
                num_inner_block_cols + tbi);
            int current_block_row = static_cast<cl_uint>(
                (data.matrix_height / config.programSettings->blockSize) -
                num_inner_block_rows + lbi);

#ifndef NDEBUG
            std::cout << "Torus " << config.programSettings->torus_row << ","
                      << config.programSettings->torus_col << " MM     "
                      << current_block_row << "," << current_block_col
                      << std::endl;
#endif

            inner_mms.push_back(
                k(Buffer_a, *Buffer_left_list[block_row % 2][lbi]->bo(),
                  *Buffer_top_list[block_row % 2][tbi]->bo(), current_block_col,
                  current_block_row, blocks_per_row));
          }
        }

        // Wait for all outer MMs to complete because the results are required
        // by the next communication phase
        for (auto &run : outer_mms) {
          run.wait();
        }

#ifndef NDEBUG
        MPI_Barrier(MPI_COMM_WORLD);
        if (is_calulating_lu_block)
          std::cout << "---------------" << std::endl;
#endif
      }
    }

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Torus " << config.programSettings->torus_row << ","
              << config.programSettings->torus_col << "End! " << std::endl;

#ifndef NDEBUG
    std::cout << "Torus " << config.programSettings->torus_row << ","
              << config.programSettings->torus_col
              << "Wait time: " << currentwaittime.count() << "s" << std::endl;
    std::cout << "Torus " << config.programSettings->torus_row << ","
              << config.programSettings->torus_col << " Exit    " << i
              << std::endl;
#endif

    std::chrono::duration<double> timespan =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    gefaExecutionTimes.push_back(timespan.count());

    // Execute GESL
    t1 = std::chrono::high_resolution_clock::now();
    t2 = std::chrono::high_resolution_clock::now();
    timespan =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    geslExecutionTimes.push_back(timespan.count());
  }

  /* --- Read back results from Device --- */

  Buffer_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (!config.programSettings->isDiagonallyDominant) {
    Buffer_pivot.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  std::unique_ptr<linpack::LinpackExecutionTimings> results(
      new linpack::LinpackExecutionTimings{gefaExecutionTimes,
                                           geslExecutionTimes});

  MPI_Barrier(MPI_COMM_WORLD);

  return results;
}

} // namespace accl_buffers
} // namespace execution
} // namespace linpack

#endif
