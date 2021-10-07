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
#ifndef EXECUTION_TYPES_EXECUTION_PCIE_HPP
#define EXECUTION_TYPES_EXECUTION_PCIE_HPP

/* C++ standard library headers */
#include <chrono>
#include <fstream>
#include <memory>
#include <vector>
#include <list>

/* External library headers */
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif

#include "parameters.h"
#include "linpack_benchmark.hpp"

namespace linpack {
namespace execution {
namespace pcie {

/*
 Prepare kernels and execute benchmark

 @copydoc bm_execution::calculate()
*/
std::unique_ptr<linpack::LinpackExecutionTimings>
calculate(const hpcc_base::ExecutionSettings<linpack::LinpackProgramSettings>&config,
          HOST_DATA_TYPE* A,
          HOST_DATA_TYPE* b,
          cl_int* ipvt) {

    int err;

    uint blocks_per_row = config.programSettings->matrixSize / config.programSettings->blockSize;

    // Communicate with all ranks in the same row of the torus
    MPI_Comm row_communicator;
    MPI_Comm col_communicator;

    MPI_Comm_split(MPI_COMM_WORLD, config.programSettings->torus_row, 0, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, config.programSettings->torus_col, 0, &col_communicator);

    cl::CommandQueue buffer_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)

    // Create Buffers for input and output
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_b(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize);
    cl::Buffer Buffer_pivot(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int)*config.programSettings->matrixSize);

    // Buffers only used to store data received over the network layer
    // The content will not be modified by the host
    cl::Buffer Buffer_lu1(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize)*(config.programSettings->blockSize));
    cl::Buffer Buffer_lu2(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize)*(config.programSettings->blockSize));
    cl::Buffer Buffer_top(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
    cl::Buffer Buffer_left(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
    cl::Buffer Buffer_network_scaling(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize));

    /* --- Setup MPI communication and required additional buffers --- */

    HOST_DATA_TYPE *lu_block, *lu_trans_block;
    posix_memalign(reinterpret_cast<void**>(&lu_block), 1024, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
    posix_memalign(reinterpret_cast<void**>(&lu_trans_block), 1024, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));

    std::vector<HOST_DATA_TYPE*> left_blocks(blocks_per_row);
    std::vector<HOST_DATA_TYPE*> top_blocks(blocks_per_row);

    for (int i =0; i < blocks_per_row; i++) {
        posix_memalign(reinterpret_cast<void**>(&left_blocks[i]), 1024, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
        posix_memalign(reinterpret_cast<void**>(&top_blocks[i]), 1024, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
    }

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> gefaExecutionTimes;
    std::vector<double> geslExecutionTimes;
    std::vector<double> gefaWaitTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {

        err = buffer_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
        ASSERT_CL(err)
        err = buffer_queue.enqueueWriteBuffer(Buffer_b, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
        ASSERT_CL(err)
        buffer_queue.finish();

        // Command queues 
        // A new command queue is created for every iteration of the algorithm to reduce the overhead
        // of too large queues
        std::list<cl::CommandQueue> lu_queues;
        std::list<cl::CommandQueue> top_queues;
        std::list<cl::CommandQueue> left_queues;
        std::list<std::vector<cl::Buffer>> left_buffers;
        std::list<std::vector<cl::Buffer>> top_buffers;
        std::list<std::vector<cl::CommandQueue>> inner_queues;
        std::list<std::vector<cl::Kernel>> kernels;

        // User event that is used to start actual execution of benchmark kernels
        cl::UserEvent start_event(*config.context, &err);
        ASSERT_CL(err);
        std::list<std::vector<cl::Event>> all_events;
        all_events.emplace_back();
        all_events.back().emplace_back(start_event);
        all_events.emplace_back();

        left_buffers.emplace_back();
        top_buffers.emplace_back();
        kernels.emplace_back();
        inner_queues.emplace_back();
        for (uint rep = 0; rep < config.programSettings->kernelReplications; rep++) {
            inner_queues.back().emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2, twait1, twait2;
        std::chrono::duration<double> currentwaittime = std::chrono::duration<double>::zero();

        uint current_replication = 0;

        std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  "Start! " << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = std::chrono::high_resolution_clock::now();
        // Trigger the user event that will start the first tasks in the queue
        start_event.setStatus(CL_COMPLETE);

        // For every row of blocks create kernels and enqueue them
        for (int block_row=0; block_row < config.programSettings->matrixSize / config.programSettings->blockSize * config.programSettings->torus_width; block_row++) {

            // Create Command queues
            lu_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            top_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            left_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)

            // already emplace new buffer list for next iteration since left and top buffers need to be stored until all MMs are executed.
            // this is only the case after the next iteration is finished, because the inner MMs are calculated overlapped with the next iteration!
            left_buffers.emplace_back();
            top_buffers.emplace_back();

            int local_block_row_remainder = (block_row % config.programSettings->torus_width);
            int local_block_row= (block_row / config.programSettings->torus_width);
            bool in_same_row_as_lu = local_block_row_remainder == config.programSettings->torus_row;
            bool in_same_col_as_lu = local_block_row_remainder == config.programSettings->torus_col;
            int start_row_index = local_block_row + ((local_block_row_remainder >= config.programSettings->torus_row) ? 1: 0); 
            int start_col_index = local_block_row + ((local_block_row_remainder >= config.programSettings->torus_col) ? 1: 0);
            int num_left_blocks = (in_same_col_as_lu) ? blocks_per_row - start_row_index : 0;
            int num_top_blocks = (in_same_row_as_lu) ? blocks_per_row - start_col_index : 0;
            int num_inner_block_rows = (blocks_per_row - start_row_index);
            int num_inner_block_cols = (num_inner_block_rows > 0) ? (blocks_per_row - start_col_index) : 0;
            num_inner_block_rows = (num_inner_block_cols > 0) ?num_inner_block_rows : 0;
            int num_network_layer_executions = (config.programSettings->matrixSize / config.programSettings->blockSize) - std::min(start_col_index, start_row_index);
            num_network_layer_executions = std::max(num_network_layer_executions, 1);
            std::vector<cl_uint> network_layer_op_flags(num_network_layer_executions);
            std::fill(network_layer_op_flags.begin(), network_layer_op_flags.end(), 0);
            bool is_calulating_lu_block = (in_same_col_as_lu && in_same_row_as_lu);

            if (is_calulating_lu_block) {
                // create the LU kernel
                kernels.back().emplace_back(*config.program, "lu",
                                            &err);
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " LU     " << local_block_row << "," << local_block_row <<  std::endl;
#endif
                err = kernels.back().back().setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(1, Buffer_lu1);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(2, Buffer_lu2);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(3, local_block_row);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(4, local_block_row);
                ASSERT_CL(err)
                err =kernels.back().back().setArg(5, config.programSettings->matrixSize / config.programSettings->blockSize);
                ASSERT_CL(err)
                all_events.back().emplace_back();
                err = lu_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err)
                // read back result of LU calculation so it can be distributed 
                err = lu_queues.back().enqueueReadBuffer(Buffer_lu2, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, lu_block);
                ASSERT_CL(err)
                err = lu_queues.back().enqueueReadBuffer(Buffer_lu1, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, lu_trans_block, NULL, &all_events.back().back());
                ASSERT_CL(err)
            }

            // Exchange LU blocks on all ranks to prevent stalls in MPI broadcast
            // All tasks until now need to be executed so we can use the result of the LU factorization and communicate it via MPI with the other FPGAs
            lu_queues.back().finish();

            // Broadcast LU block in column to update all left blocks
            MPI_Bcast(lu_block, config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, col_communicator);
            // Broadcast LU block in row to update all top blocks
            MPI_Bcast(lu_trans_block, config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, row_communicator);

            if (num_top_blocks > 0) {

                // Copy LU block to FPGA for calulation of top blocks only if required
                err = top_queues.back().enqueueWriteBuffer(Buffer_lu1, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, lu_trans_block);
                ASSERT_CL(err)

                // Create top kernels
                for (int tops=start_col_index; tops < (config.programSettings->matrixSize / config.programSettings->blockSize); tops++) {
                    kernels.back().emplace_back(*config.program, "top_update",
                                                    &err);
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Top    " << local_block_row << "," << tops <<  std::endl;
#endif
                    ASSERT_CL(err);     
                    err = kernels.back().back().setArg(0, Buffer_a);
                    ASSERT_CL(err);    
                    err = kernels.back().back().setArg(1, Buffer_top);
                    ASSERT_CL(err);    
                    err = kernels.back().back().setArg(2, Buffer_lu1);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(3, (tops == start_col_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(4, tops);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(5, local_block_row);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(6, config.programSettings->matrixSize / config.programSettings->blockSize);
                    ASSERT_CL(err)

                    err = top_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err) 

                    if (tops + 1 == (config.programSettings->matrixSize / config.programSettings->blockSize)) {
                        all_events.back().emplace_back();
                        err = top_queues.back().enqueueReadBuffer(Buffer_top, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, top_blocks[tops - start_col_index],
                                     &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                        ASSERT_CL(err) 
                    }
                    else {
                        err = top_queues.back().enqueueReadBuffer(Buffer_top, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, top_blocks[tops - start_col_index]);
                        ASSERT_CL(err)
                    }

                }
            }
            if (num_left_blocks > 0) {

                // Copy LU block to FPGA for calulation of left blocks only if required
                err = left_queues.back().enqueueWriteBuffer(Buffer_lu2, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, lu_block);
                ASSERT_CL(err)
                // Create left kernels
                for (int tops=start_row_index; tops < (config.programSettings->matrixSize / config.programSettings->blockSize); tops++) {
                    kernels.back().emplace_back(*config.program, "left_update",
                                                    &err);
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  " Left   " <<tops  << "," << local_block_row <<  std::endl;
#endif
                    ASSERT_CL(err);     
                    err = kernels.back().back().setArg(0, Buffer_a);
                    ASSERT_CL(err);    
                    err = kernels.back().back().setArg(1, Buffer_left);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(2, Buffer_lu2);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(3, (tops == start_row_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(4, local_block_row);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(5, tops);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(6, config.programSettings->matrixSize / config.programSettings->blockSize);
                    ASSERT_CL(err)

                    err = left_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err) 

                    if (tops + 1 == (config.programSettings->matrixSize / config.programSettings->blockSize)) {
                        all_events.back().emplace_back();
                        err = left_queues.back().enqueueReadBuffer(Buffer_left, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, left_blocks[tops - start_row_index],
                                     &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                        ASSERT_CL(err) 
                    }
                    else {
                        err = left_queues.back().enqueueReadBuffer(Buffer_left, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, left_blocks[tops - start_row_index]);
                        ASSERT_CL(err) 
                    }
                }
            }
            // Wait until all top and left blocks are calculated
            top_queues.back().finish();
            left_queues.back().finish();

            // Send the left and top blocks to all other ranks so they can be used to update all inner blocks
            for (int lbi=0; lbi < blocks_per_row - local_block_row; lbi++) {
                MPI_Bcast(left_blocks[lbi], config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, row_communicator);
            }
            for (int tbi=0; tbi < blocks_per_row  - local_block_row; tbi++) {
                MPI_Bcast(top_blocks[tbi], config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, col_communicator);
            }

            // update all remaining inner blocks using only global memory

            all_events.emplace_back();
            //auto communication_events = all_events.back();

            // Write all left and top blocks to FPGA memory
            for (int lbi=0; lbi < num_inner_block_rows; lbi++) {
                left_buffers.back().emplace_back(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
                err = inner_queues.back()[0].enqueueWriteBuffer(left_buffers.back().back(), CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, left_blocks[lbi]);
            }
            for (int tbi=0; tbi < num_inner_block_cols; tbi++) {
                top_buffers.back().emplace_back(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
                err = inner_queues.back()[0].enqueueWriteBuffer(top_buffers.back().back(), CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize, top_blocks[tbi]);
            }

            uint current_update = 0;
            uint total_inner_updates_first_row = top_buffers.back().size();
            uint updates_per_replication = total_inner_updates_first_row / config.programSettings->kernelReplications;
            uint total_inner_updates = (top_buffers.back().size() - 1) * (left_buffers.back().size() - 1);
            uint total_updates_per_replication = total_inner_updates/ config.programSettings->kernelReplications;

            // Wait until data is copied to FPGA
            inner_queues.back()[0].finish();

            for (auto l = std::next(left_buffers.back().begin()); l < left_buffers.back().end(); l++) {
                // select the matrix multiplication kernel that should be used for this block updated 
                kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                    &err);

                int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols);
                int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows + std::distance(left_buffers.back().begin(), l));  
                ASSERT_CL(err);
                err = kernels.back().back().setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(1, *l);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(2, *top_buffers.back().begin());
                ASSERT_CL(err)
                err = kernels.back().back().setArg(3, block_col);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(4, block_row);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(5, blocks_per_row);
                ASSERT_CL(err)

                if ((left_buffers.back().size() - 1) - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner L Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                    // this is the last taks that will be enqueued in this queue, so create an event
                    all_events.back().emplace_back();
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));         
                    //err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &communication_events, &(all_events.back().back()));         
                }
                else {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner L " << block_row << "," << block_col <<  std::endl;
#endif 
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
                    //err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &communication_events);         
                }
                current_update++;
                current_replication = (current_replication + 1) % config.programSettings->kernelReplications;
            }

            current_update = 0;
            for (auto t = top_buffers.back().begin(); t < top_buffers.back().end(); t++) {
                // select the matrix multiplication kernel that should be used for this block updated 
                kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                    &err);

                int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols + std::distance(top_buffers.back().begin(), t));
                int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows);

                ASSERT_CL(err);
                err = kernels.back().back().setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(1, *left_buffers.back().begin());
                ASSERT_CL(err)
                err = kernels.back().back().setArg(2, *t);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(3, block_col);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(4, block_row);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(5, blocks_per_row);
                ASSERT_CL(err)
                // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                if (top_buffers.back().size() - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner T Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                    // this is the last taks that will be enqueued in this queue, so create an event
                    all_events.back().emplace_back();
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));         
                }
                else {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner T " << block_row << "," << block_col <<  std::endl;
#endif 
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
                }
                ASSERT_CL(err) 
                current_update++;
                current_replication = (current_replication + 1) % config.programSettings->kernelReplications;
            }
            
            // count the inner MM already to next iteration by creating new buffers in the queue
            all_events.emplace_back();
            kernels.emplace_back();
            inner_queues.emplace_back();
            current_update = 0;
            for (uint rep = 0; rep < config.programSettings->kernelReplications; rep++) {
                inner_queues.back().emplace_back(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)
            }

            for (auto l = std::next(left_buffers.back().begin()); l < left_buffers.back().end(); l++) {
                for (auto t = std::next(top_buffers.back().begin()); t < top_buffers.back().end(); t++) {
                    // select the matrix multiplication kernel that should be used for this block updated 
                    kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                        &err);

                    int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols + std::distance(top_buffers.back().begin(), t));
                    int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows + std::distance(left_buffers.back().begin(), l));
  
                    ASSERT_CL(err);
                    err = kernels.back().back().setArg(0, Buffer_a);
                    ASSERT_CL(err);
                    err = kernels.back().back().setArg(1, *l);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(2, *t);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(3, block_col);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(4, block_row);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(5, blocks_per_row);
                    ASSERT_CL(err)

                    // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                    if (((top_buffers.back().size() - 1) * (left_buffers.back().size() - 1)) - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                        // this is the last taks that will be enqueued in this queue, so create an event
                        all_events.back().emplace_back();
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(std::prev(all_events.end())))), &(all_events.back().back()));         
                    }
                    else {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
#endif 
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[(current_replication)].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(std::prev(all_events.end())))));         
                    }

                    ASSERT_CL(err)
                    current_update++;
                    current_replication = (current_replication + 1) % config.programSettings->kernelReplications;
                }
            }
#ifndef NDEBUG
            MPI_Barrier(MPI_COMM_WORLD);
            if (is_calulating_lu_block) std::cout << "---------------" << std::endl;

            // // // Execute GEFA
            // if (block_row == 0) {
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     t1 = std::chrono::high_resolution_clock::now();
            //     // Trigger the user event that will start the first tasks in the queue
            //     start_event.setStatus(CL_COMPLETE);
            // }
#endif

#ifndef NDEBUG
            cl::Event::waitForEvents(all_events.back());
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Done    " << block_row <<  std::endl;

            if (block_row == blocks_per_row * config.programSettings->torus_width - 1) {
                // wait until the last LU queue is done since it will be the last required operation
                lu_queues.back().finish();
                t2 = std::chrono::high_resolution_clock::now();

                // Finish all other queues
                top_queues.back().finish();
                left_queues.back().finish();
                cl::Event::waitForEvents(all_events.back());

            }
#endif
        }
#ifdef NDEBUG
        int count = 0;
        for (auto evs : all_events) {
            count++;
            cl::Event::waitForEvents(evs);
            // std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  "Step " << count << " of " << all_events.size() << std::endl;
        }
        lu_queues.back().finish();
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  "End! " << std::endl;
#endif

#ifndef NDEBUG
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  "Wait time: " << currentwaittime.count() << "s" << std::endl;
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Exit    " << i <<  std::endl;
#endif

        std::chrono::duration<double> timespan =
                std::chrono::duration_cast<std::chrono::duration<double>>
                                                                    (t2 - t1);
        gefaExecutionTimes.push_back(timespan.count());

        // Execute GESL
        t1 = std::chrono::high_resolution_clock::now();
        // lu_queue.enqueueTask(geslkernel);
        // lu_queue.finish();
        t2 = std::chrono::high_resolution_clock::now();
        timespan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        geslExecutionTimes.push_back(timespan.count());
    }

    /* --- Read back results from Device --- */

#ifdef USE_SVM
    err = clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(A), 0,
                        NULL, NULL);
    ASSERT_CL(err)
    err = clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(b), 0,
                        NULL, NULL);
    ASSERT_CL(err)
    err = clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(ipvt), 0,
                        NULL, NULL);
    ASSERT_CL(err)
    
    // read back result from temporary buffer
    for (int k=0; k < config.programSettings->matrixSize * config.programSettings->matrixSize; k++) {
        A[k] = A_tmp[k];
    }
    clSVMFree((*config.context)(), reinterpret_cast<void*>(A_tmp));

#else
    buffer_queue.enqueueReadBuffer(Buffer_a, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
    // buffer_queue.enqueueReadBuffer(Buffer_b, CL_TRUE, 0,
    //                                  sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
    if (!config.programSettings->isDiagonallyDominant) {
        buffer_queue.enqueueReadBuffer(Buffer_pivot, CL_TRUE, 0,
                                        sizeof(cl_int)*config.programSettings->matrixSize, ipvt);
    }
#endif

    /* --- Clean up MPI communication buffers --- */
    free(lu_block);
    free(lu_trans_block);

    for (int i =0; i < left_blocks.size(); i++) {
        free(top_blocks[i]);
        free(left_blocks[i]);
    }

    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&col_communicator);

    std::unique_ptr<linpack::LinpackExecutionTimings> results(
                    new linpack::LinpackExecutionTimings{gefaExecutionTimes, geslExecutionTimes});
    
    MPI_Barrier(MPI_COMM_WORLD);

    return results;
}

}   // namespace pcie
}   // namespace execution
}  // namespace linpack

#endif