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
#include <thread>

/* External library headers */
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif
#ifdef _OPENMP
#include "omp.h"
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
std::map<std::string, std::vector<double>>
calculate(const hpcc_base::ExecutionSettings<linpack::LinpackProgramSettings>&config,
          linpack::LinpackData& data) {

    cl_int err;

    int num_omp_threads = 1;
#ifdef _OPENMP
    num_omp_threads = omp_get_num_threads();
#endif

    uint blocks_per_row = data.matrix_width / config.programSettings->blockSize;
    uint blocks_per_col = data.matrix_height / config.programSettings->blockSize;

    // Communicate with all ranks in the same row of the torus
    MPI_Comm row_communicator;
    MPI_Comm col_communicator;

    MPI_Comm_split(MPI_COMM_WORLD, config.programSettings->torus_row, 0, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, config.programSettings->torus_col, 0, &col_communicator);

    cl::CommandQueue buffer_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)

    // Create Buffers for input and output
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*data.matrix_height * data.matrix_width);
    cl::Buffer Buffer_b(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*data.matrix_width);
    cl::Buffer Buffer_pivot(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int)*data.matrix_height);


    /* --- Setup MPI communication and required additional buffers --- */
    HOST_DATA_TYPE *lu_block, *lu_trans_block;
    posix_memalign(reinterpret_cast<void**>(&lu_block), 4096, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
    posix_memalign(reinterpret_cast<void**>(&lu_trans_block), 4096, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));

    // Buffers only used to store data received over the network layer
    // The content will not be modified by the host
    cl::Buffer Buffer_lu1(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize)*(config.programSettings->blockSize));
    cl::Buffer Buffer_lu2(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize)*(config.programSettings->blockSize));

    std::vector<cl::Buffer> Buffer_left_list;
    std::vector<cl::Buffer> Buffer_top_list;
    std::vector<HOST_DATA_TYPE*> left_blocks(blocks_per_col);
    std::vector<HOST_DATA_TYPE*> top_blocks(blocks_per_row);

    for (int i =0; i < blocks_per_row; i++) {
        posix_memalign(reinterpret_cast<void**>(&(top_blocks[i])), 4096, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
        Buffer_top_list.emplace_back(*config.context, CL_MEM_WRITE_ONLY,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
    }

    for (int i =0; i < blocks_per_col; i++) {
        posix_memalign(reinterpret_cast<void**>(&(left_blocks[i])), 4096, sizeof(HOST_DATA_TYPE) * (config.programSettings->blockSize)*(config.programSettings->blockSize));
        Buffer_left_list.emplace_back(*config.context, CL_MEM_WRITE_ONLY,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
    }

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> gefaExecutionTimes;
    std::vector<double> geslExecutionTimes;
    std::vector<double> gefaWaitTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {

        err = buffer_queue.enqueueWriteBuffer(Buffer_a, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*data.matrix_height*data.matrix_width, data.A);
        ASSERT_CL(err)
        err = buffer_queue.enqueueWriteBuffer(Buffer_b, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)* data.matrix_width, data.b);
        ASSERT_CL(err)
        buffer_queue.finish();

        // Command queues 
        // A new command queue is created for every iteration of the algorithm to reduce the overhead
        // of too large queues
        std::deque<cl::CommandQueue> lu_queues;
        std::deque<cl::CommandQueue> top_queues;
        std::deque<cl::CommandQueue> left_queues;
        std::deque<std::vector<cl::Buffer>> left_buffers;
        std::deque<std::vector<cl::Buffer>> top_buffers;
        std::deque<std::vector<cl::CommandQueue>> inner_queues;
        std::deque<std::vector<cl::Kernel>> kernels;
        std::thread flush_thread;

        // User event that is used to start actual execution of benchmark kernels
        cl::UserEvent start_event(*config.context, &err);
        ASSERT_CL(err);
        std::deque<std::vector<cl::Event>> all_events;
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

        std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  "Start! " << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = std::chrono::high_resolution_clock::now();
        // Trigger the user event that will start the first tasks in the queue
        start_event.setStatus(CL_COMPLETE);


        int kernel_offset = 0;
        #pragma omp parallel
        {

        #pragma omp single
        all_events.back().reserve(num_omp_threads*config.programSettings->kernelReplications*3);
        uint current_replication = 0;

        // For every row of blocks create kernels and enqueue them
        for (int block_row=0; block_row < config.programSettings->matrixSize / config.programSettings->blockSize; block_row++) {

            int local_block_row_remainder = (block_row % config.programSettings->torus_height);
            int local_block_row = (block_row / config.programSettings->torus_height);
            int local_block_col_remainder = (block_row % config.programSettings->torus_width);
            int local_block_col = (block_row / config.programSettings->torus_width);
            bool in_same_row_as_lu = local_block_row_remainder == config.programSettings->torus_row;
            bool in_same_col_as_lu = local_block_col_remainder == config.programSettings->torus_col;
            int start_row_index = local_block_row + ((local_block_row_remainder >= config.programSettings->torus_row) ? 1: 0); 
            int start_col_index = local_block_col + ((local_block_col_remainder >= config.programSettings->torus_col) ? 1: 0);
            int num_left_blocks = (in_same_col_as_lu) ? blocks_per_col - start_row_index : 0;
            int num_top_blocks = (in_same_row_as_lu) ? blocks_per_row - start_col_index : 0;
            int num_inner_block_rows = (blocks_per_col - start_row_index);
            int num_inner_block_cols = (num_inner_block_rows > 0) ? (blocks_per_row - start_col_index) : 0;
            num_inner_block_rows = (num_inner_block_cols > 0) ?num_inner_block_rows : 0;
            bool is_calulating_lu_block = (in_same_col_as_lu && in_same_row_as_lu);

#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Start iteration     " << block_row <<  std::endl;
#endif

            uint total_inner_updates_first_row = num_inner_block_cols;
            uint updates_per_replication = total_inner_updates_first_row / config.programSettings->kernelReplications;
            uint total_inner_updates = (num_inner_block_cols - 1) * (num_inner_block_rows - 1);
            uint total_updates_per_replication = total_inner_updates/ config.programSettings->kernelReplications;
            uint current_update = 0;

            std::vector<cl::Kernel> private_kernels;


            #pragma omp single
            {

            // Create Command queues
            lu_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            top_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            left_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)

            if (is_calulating_lu_block) {
                // create the LU kernel
                private_kernels.emplace_back(*config.program, "lu",
                                            &err);
                ASSERT_CL(err);
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " LU     " << local_block_row << "," << local_block_col <<  std::endl;
#endif
                err = private_kernels.back().setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = private_kernels.back().setArg(1, Buffer_lu1);
                ASSERT_CL(err);
                err = private_kernels.back().setArg(2, Buffer_lu2);
                ASSERT_CL(err);
                err = private_kernels.back().setArg(3, local_block_col);
                ASSERT_CL(err)
                err = private_kernels.back().setArg(4, local_block_row);
                ASSERT_CL(err)
                err =private_kernels.back().setArg(5, blocks_per_row);
                ASSERT_CL(err)
                err = lu_queues.back().enqueueNDRangeKernel(private_kernels.back(), cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err)
                // read back result of LU calculation so it can be distributed 
                err = lu_queues.back().enqueueReadBuffer(Buffer_lu2, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), lu_block);
                ASSERT_CL(err)
                err = lu_queues.back().enqueueReadBuffer(Buffer_lu1, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), lu_trans_block);
                ASSERT_CL(err)
            }

            // Exchange LU blocks on all ranks to prevent stalls in MPI broadcast
            // All tasks until now need to be executed so we can use the result of the LU factorization and communicate it via MPI with the other FPGAs
            lu_queues.back().finish();

            // Broadcast LU block in column to update all left blocks
            MPI_Bcast(lu_block, config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, col_communicator);
            // Broadcast LU block in row to update all top blocks
            MPI_Bcast(lu_trans_block, config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_col_remainder, row_communicator);
           }

            if (num_top_blocks > 0) {

                #pragma omp single
                {
                cl::Event write_lu_trans_done;
                // Copy LU block to FPGA for calulation of top blocks only if required
                err = top_queues.back().enqueueWriteBuffer(Buffer_lu1, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), lu_trans_block, NULL, &write_lu_trans_done);
                ASSERT_CL(err)
                (*std::prev(std::prev(all_events.end()))).push_back(write_lu_trans_done);
                }

                // Create top kernels
                #pragma omp for
                for (int tops=start_col_index; tops < blocks_per_row; tops++) {
                    cl::Kernel k(*config.program, "top_update",
                                                    &err);
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Top    " << local_block_row << "," << tops <<  std::endl;
#endif
                    ASSERT_CL(err);     
                    err = k.setArg(0, Buffer_a);
                    ASSERT_CL(err);    
                    err = k.setArg(1, Buffer_top_list[tops - start_col_index]);
                    ASSERT_CL(err);    
                    err = k.setArg(2, Buffer_lu1);
                    ASSERT_CL(err) 
                    err = k.setArg(3, (tops == start_col_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = k.setArg(4, tops);
                    ASSERT_CL(err)
                    err = k.setArg(5, local_block_row);
                    ASSERT_CL(err)
                    err = k.setArg(6, blocks_per_row);
                    ASSERT_CL(err)

                    err = top_queues.back().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err) 

                    err = top_queues.back().enqueueReadBuffer(Buffer_top_list[tops - start_col_index], CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), top_blocks[tops - start_col_index]);
                    ASSERT_CL(err)

                    private_kernels.push_back(k);

                }
            }
            if (num_left_blocks > 0) {

                #pragma omp single
                {
                cl::Event write_lu_done;
                // Copy LU block to FPGA for calulation of left blocks only if required
                err = left_queues.back().enqueueWriteBuffer(Buffer_lu2, CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), lu_block, NULL, &write_lu_done);
                ASSERT_CL(err)
                (*std::prev(std::prev(all_events.end()))).push_back(write_lu_done);
                }

                // Create left kernels
                #pragma omp for
                for (int tops=start_row_index; tops < blocks_per_col; tops++) {
                    cl::Kernel k(*config.program, "left_update",
                                                    &err);
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  " Left   " <<tops  << "," << local_block_col <<  std::endl;
#endif
                    ASSERT_CL(err);     
                    err = k.setArg(0, Buffer_a);
                    ASSERT_CL(err);    
                    err = k.setArg(1, Buffer_left_list[tops - start_row_index]);
                    ASSERT_CL(err) 
                    err = k.setArg(2, Buffer_lu2);
                    ASSERT_CL(err) 
                    err = k.setArg(3, (tops == start_row_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = k.setArg(4, local_block_col);
                    ASSERT_CL(err)
                    err = k.setArg(5, tops);
                    ASSERT_CL(err)
                    err = k.setArg(6, blocks_per_row);
                    ASSERT_CL(err)

                    err = left_queues.back().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err) 

                    err = left_queues.back().enqueueReadBuffer(Buffer_left_list[tops - start_row_index], CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), left_blocks[tops - start_row_index]);

                    ASSERT_CL(err) 

                    private_kernels.push_back(k);
                }
            }

            #pragma omp single
            {
            // Wait until all top and left blocks are calculated
            top_queues.back().finish();
            left_queues.back().finish();

            // Send the left and top blocks to all other ranks so they can be used to update all inner blocks
            for (int lbi=0; lbi < std::max(static_cast<int>(blocks_per_col - local_block_col), 0); lbi++) {
                MPI_Bcast(left_blocks[lbi], config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_col_remainder, row_communicator);
            }
            for (int tbi=0; tbi < std::max(static_cast<int>(blocks_per_row  - local_block_row), 0); tbi++) {
                MPI_Bcast(top_blocks[tbi], config.programSettings->blockSize*config.programSettings->blockSize, MPI_DATA_TYPE, local_block_row_remainder, col_communicator);
            }

            // update all remaining inner blocks using only global memory

            // all_events.emplace_back();
            //auto communication_events = all_events.back();
            left_buffers.emplace_back();
            top_buffers.emplace_back();
            
            cl::CommandQueue buffer_transfer_queue(*config.context, *config.device, 0, &err);

            // Write all left and top blocks to FPGA memory
            for (int lbi=0; lbi < num_inner_block_rows; lbi++) {
                left_buffers.back().emplace_back(*config.context, CL_MEM_READ_ONLY,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize));
                err = buffer_transfer_queue.enqueueWriteBuffer(left_buffers.back().back(), CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), left_blocks[lbi]);
            }
            for (int tbi=0; tbi < num_inner_block_cols; tbi++) {
                top_buffers.back().emplace_back(*config.context, CL_MEM_READ_ONLY,
                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * config.programSettings->blockSize);
                err = buffer_transfer_queue.enqueueWriteBuffer(top_buffers.back().back(), CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize * (config.programSettings->blockSize), top_blocks[tbi]);
            }

            kernel_offset = kernels.back().size();
            kernels.back().resize(std::max(kernel_offset + num_inner_block_rows - 1 + num_inner_block_cols,0));

            all_events.emplace_back();
            all_events.back().reserve(num_omp_threads*config.programSettings->kernelReplications*2);

            // Wait until data is copied to FPGA
            buffer_transfer_queue.finish();
            }
            current_update = 0;    

            #pragma omp for
            for (int lbi=1; lbi < num_inner_block_rows; lbi++) {

                current_replication = (lbi)  % config.programSettings->kernelReplications;

                // select the matrix multiplication kernel that should be used for this block updated 
#ifdef INTEL_FPGA
                cl::Kernel k(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                    &err);
#endif
#ifdef XILINX_FPGA
                cl::Kernel k(*config.program, ("inner_update_mm0:{inner_update_mm0_" + std::to_string(current_replication + 1) + "}").c_str(),
                                    &err);
#endif

                int block_col = static_cast<cl_uint>((data.matrix_width / config.programSettings->blockSize) - num_inner_block_cols);
                int block_row = static_cast<cl_uint>((data.matrix_height / config.programSettings->blockSize) - num_inner_block_rows + lbi);  
                ASSERT_CL(err);
                err = k.setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = k.setArg(1, left_buffers.back()[lbi]);
                ASSERT_CL(err)
                err = k.setArg(2, top_buffers.back()[0]);
                ASSERT_CL(err)
                err = k.setArg(3, block_col);
                ASSERT_CL(err)
                err = k.setArg(4, block_row);
                ASSERT_CL(err)
                err = k.setArg(5, blocks_per_row);
                ASSERT_CL(err)

                // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                if ((num_inner_block_rows - 1)/num_omp_threads - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner L Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                    // this is the last taks that will be enqueued in this queue, so create an event
                    cl::Event ev;
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))), &ev);  

                    #pragma omp critical
                    all_events.back().push_back(ev);           
                }
                else {
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner L " << block_row << "," << block_col <<  std::endl;
#endif 
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));         
                }

                kernels.back()[kernel_offset + lbi - 1] = k;

                current_update++;
            }

            current_update = 0;
            #pragma omp for
            for (int tbi=0; tbi < num_inner_block_cols; tbi++) {

                current_replication = (tbi)  % config.programSettings->kernelReplications;

                // select the matrix multiplication kernel that should be used for this block updated 
#ifdef INTEL_FPGA
                cl::Kernel k(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                    &err);
#endif
#ifdef XILINX_FPGA
                cl::Kernel k(*config.program, ("inner_update_mm0:{inner_update_mm0_" + std::to_string(current_replication + 1) + "}").c_str(),
                                    &err);
#endif
                int block_col = static_cast<cl_uint>((data.matrix_width / config.programSettings->blockSize) - num_inner_block_cols + tbi);
                int block_row = static_cast<cl_uint>((data.matrix_height / config.programSettings->blockSize) - num_inner_block_rows);

                ASSERT_CL(err);
                err = k.setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = k.setArg(1, left_buffers.back()[0]);
                ASSERT_CL(err)
                err = k.setArg(2, top_buffers.back()[tbi]);
                ASSERT_CL(err)
                err = k.setArg(3, block_col);
                ASSERT_CL(err)
                err = k.setArg(4, block_row);
                ASSERT_CL(err)
                err = k.setArg(5, blocks_per_row);
                ASSERT_CL(err)
                // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                if ((num_inner_block_cols)/num_omp_threads - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                    // this is the last taks that will be enqueued in this queue, so create an event
                    cl::Event ev;
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))), &ev);  

                    #pragma omp critical
                    all_events.back().push_back(ev);           
                }
                else {
#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
#endif 
                    // Distribute the workload over all available matrix multiplication kernels
                    err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));         
                }
                ASSERT_CL(err) 
                kernels.back()[kernel_offset + tbi + num_inner_block_rows - 1] = k;

                current_update++;
            }
            
            #pragma omp single
            {
            // count the inner MM already to next iteration by creating new buffers in the queue
            all_events.emplace_back();
            all_events.back().reserve(num_omp_threads*config.programSettings->kernelReplications);
            kernels.emplace_back(total_inner_updates);
            inner_queues.emplace_back();
            current_update = 0;
            for (uint rep = 0; rep < config.programSettings->kernelReplications; rep++) {
                inner_queues.back().emplace_back(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)
            }

            }

            #pragma omp for collapse(2) schedule(static)
            for (int lbi=1; lbi < num_inner_block_rows; lbi++) {
                for (int tbi=1; tbi < num_inner_block_cols; tbi++) {
                    // select the matrix multiplication kernel that should be used for this block updated 

                    current_replication = (lbi * num_inner_block_cols + tbi)  % config.programSettings->kernelReplications;

#ifdef INTEL_FPGA
                    cl::Kernel k(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                        &err);
#endif
#ifdef XILINX_FPGA
                    cl::Kernel k(*config.program, ("inner_update_mm0:{inner_update_mm0_" + std::to_string(current_replication + 1) + "}").c_str(),
                                        &err);
#endif

                    int block_col = static_cast<cl_uint>((data.matrix_width / config.programSettings->blockSize) - num_inner_block_cols + tbi);
                    int block_row = static_cast<cl_uint>((data.matrix_height / config.programSettings->blockSize) - num_inner_block_rows + lbi);
  
                    ASSERT_CL(err);
                    err = k.setArg(0, Buffer_a);
                    ASSERT_CL(err);
                    err = k.setArg(1, left_buffers.back()[lbi]);
                    ASSERT_CL(err)
                    err = k.setArg(2, top_buffers.back()[tbi]);
                    ASSERT_CL(err)
                    err = k.setArg(3, block_col);
                    ASSERT_CL(err)
                    err = k.setArg(4, block_row);
                    ASSERT_CL(err)
                    err = k.setArg(5, blocks_per_row);
                    ASSERT_CL(err)

                    // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                    if ((total_inner_updates)/num_omp_threads - current_update <= config.programSettings->kernelReplications) {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner Ev " << block_row << "," << block_col <<  std::endl;
#endif 
                        // this is the last taks that will be enqueued in this queue, so create an event
                        cl::Event ev;
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))), &(ev));

                        #pragma omp critical
                        all_events.back().push_back(ev);      
                    }
                    else {
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
#endif 
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[current_replication].enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1),  &(*std::prev(std::prev(all_events.end()))));
                    }

                    ASSERT_CL(err)

                    kernels.back()[(lbi-1)*(num_inner_block_cols - 1)+(tbi-1)] = k;

                    current_update++;
                }
            }

#ifdef NDEBUG
            #pragma omp single
            {
                if (flush_thread.joinable()) {
                    flush_thread.join();
                }
                // Start new thread that cuntinuously puts new tasks on the FPGA while the main thread
                // may be blocked by MPI calls
                std::thread new_thread([all_events](){ cl::Event::waitForEvents(all_events.back());});
                flush_thread.swap(new_thread);
            }
#endif

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
                t2 = std::chrono::high_resolution_clock::now();
                cl::Event::waitForEvents(all_events.back());

            }
#endif

#ifdef XILINX_FPGA
            #pragma omp single nowait
            {
            if (block_row > 2) {
                // clean up old queues and kernels
                lu_queues.pop_front();
                left_queues.pop_front();
                top_queues.pop_front();
                inner_queues.pop_front();
                left_buffers.pop_front();
                top_buffers.pop_front();
                kernels.pop_front();
                all_events.pop_front();
                all_events.pop_front();
            }
            }
#endif

        }
    }

    if (flush_thread.joinable()) {
        flush_thread.join();
    }

#ifdef NDEBUG
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
                                     sizeof(HOST_DATA_TYPE)*data.matrix_height*data.matrix_width, data.A);
    // buffer_queue.enqueueReadBuffer(Buffer_b, CL_TRUE, 0,
    //                                  sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
    if (!config.programSettings->isDiagonallyDominant) {
        buffer_queue.enqueueReadBuffer(Buffer_pivot, CL_TRUE, 0,
                                        sizeof(cl_int)*data.matrix_height, data.ipvt);
    }
    buffer_queue.finish();
#endif

    /* --- Clean up MPI communication buffers --- */
    free(lu_block);
    free(lu_trans_block);

    for (int i =0; i < left_blocks.size(); i++) {
        free(left_blocks[i]);
    }
    for (int i =0; i < top_blocks.size(); i++) {
        free(top_blocks[i]);
    }

    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&col_communicator);

    std::map<std::string, std::vector<double>> timings;
    
    timings["gefa"] = gefaExecutionTimes;
    timings["gesl"] = geslExecutionTimes;
    
    MPI_Barrier(MPI_COMM_WORLD);

    return timings;
}

}   // namespace pcie
}   // namespace execution
}  // namespace linpack

#endif