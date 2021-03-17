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

/* Related header files */
#include "execution.h"

/* C++ standard library headers */
#include <chrono>
#include <fstream>
#include <memory>
#include <vector>
#include <list>

/* External library headers */
#include "CL/cl2.hpp"
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif

namespace bm_execution {

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
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize * (config.programSettings->blockSize));
    cl::Buffer Buffer_left(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize * (config.programSettings->blockSize));
    cl::Buffer Buffer_network_scaling(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(config.programSettings->blockSize));

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
        std::list<cl::CommandQueue> network_queues_topleft;
        std::list<cl::CommandQueue> network_queues_bottomright;
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

        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2, twait1, twait2;
        std::chrono::duration<double> currentwaittime = std::chrono::duration<double>::zero();

        // For every row of blocks create kernels and enqueue them
        for (int block_row=0; block_row < config.programSettings->matrixSize / config.programSettings->blockSize * config.programSettings->torus_width; block_row++) {

            // Create Command queues
            lu_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            top_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            left_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            inner_queues.emplace_back();
            for (uint rep = 0; rep < config.programSettings->kernelReplications + 1; rep++) {
                inner_queues.back().emplace_back(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)
            }
            network_queues_bottomright.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            network_queues_topleft.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)

            left_buffers.emplace_back();
            top_buffers.emplace_back();
            all_events.emplace_back();
            kernels.emplace_back();

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
                err = kernels.back().back().setArg(1, local_block_row);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(2, local_block_row);
                ASSERT_CL(err)
                err =kernels.back().back().setArg(3, config.programSettings->matrixSize / config.programSettings->blockSize);
                ASSERT_CL(err)
                all_events.back().emplace_back();
                err = lu_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &all_events.back().back());
                ASSERT_CL(err)


                network_layer_op_flags[0] |= LU_BLOCK_OUT;
            }

            if (num_top_blocks > 0) {
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
                    err = kernels.back().back().setArg(1, Buffer_lu1);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(2, (tops == start_col_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(3, tops);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(4, local_block_row);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(5, config.programSettings->matrixSize / config.programSettings->blockSize);
                    ASSERT_CL(err)

                    if (tops + 1 == (config.programSettings->matrixSize / config.programSettings->blockSize)) {
                        all_events.back().emplace_back();
                        err = top_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                        ASSERT_CL(err) 
                    }
                    else {
                        err = top_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                        ASSERT_CL(err) 
                    }
        
                    network_layer_op_flags[0] |= TOP_BLOCK;
                    network_layer_op_flags[tops - start_col_index] |= TOP_BLOCK_OUT;

                }
            }
            if (num_left_blocks > 0) {
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
                    err = kernels.back().back().setArg(1, Buffer_lu2);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(2, (tops == start_row_index) ? CL_TRUE : CL_FALSE);
                    ASSERT_CL(err) 
                    err = kernels.back().back().setArg(3, local_block_row);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(4, tops);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(5, config.programSettings->matrixSize / config.programSettings->blockSize);
                    ASSERT_CL(err)

                    if (tops + 1 == (config.programSettings->matrixSize / config.programSettings->blockSize)) {
                        all_events.back().emplace_back();
                        err = left_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                        ASSERT_CL(err) 
                    }
                    else {
                        err = left_queues.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                        ASSERT_CL(err) 
                    }
                    network_layer_op_flags[0] |= LEFT_BLOCK;
                    network_layer_op_flags[tops - start_row_index] |= LEFT_BLOCK_OUT;
                }
            }


            uint network_forward_flags = 0;
            if (((local_block_row_remainder + config.programSettings->torus_row + 1) % config.programSettings->torus_width > 0) && (network_layer_op_flags[0] & (LEFT_BLOCK_OUT | LU_BLOCK_OUT)) && block_row + 1 !=config.programSettings->matrixSize / config.programSettings->blockSize * config.programSettings->torus_width) {
                network_forward_flags |= NETWORK_FWD_BOTTOM;
            }
            if (((local_block_row_remainder + config.programSettings->torus_row+ config.programSettings->torus_width - 1) % config.programSettings->torus_width > 0) && (num_top_blocks + num_inner_block_rows > 0)) {
                network_forward_flags |= NETWORK_FWD_TOP;
            }
            if (((local_block_row_remainder + config.programSettings->torus_col + 1) % config.programSettings->torus_width > 0) && (network_layer_op_flags[0] & (TOP_BLOCK_OUT | LU_BLOCK_OUT)) && block_row + 1 != config.programSettings->matrixSize / config.programSettings->blockSize * config.programSettings->torus_width) {
                network_forward_flags |= NETWORK_FWD_RIGHT;
            }
            if (((local_block_row_remainder + config.programSettings->torus_col + config.programSettings->torus_width - 1) % config.programSettings->torus_width > 0) && (num_left_blocks + num_inner_block_cols > 0)) {
                network_forward_flags |= NETWORK_FWD_LEFT;
            }
            // Create network kernels
            int nw_exe_count = 0;
            for (auto it = network_layer_op_flags.begin(); it < network_layer_op_flags.end(); it++) {

                uint op_flags = *it;
                bool left_block_is_received = num_inner_block_rows > nw_exe_count;
                bool top_block_is_received = num_inner_block_cols > nw_exe_count;
                if (left_block_is_received) {
                    left_buffers.back().emplace_back(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize);
                    op_flags |= STORE_LEFT_INNER;
                }
                if (top_block_is_received) {
                    top_buffers.back().emplace_back(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->blockSize*config.programSettings->blockSize);
                    op_flags |= STORE_TOP_INNER;
                }

                if (it == network_layer_op_flags.begin()) {
                    // Create the network kernel
                    kernels.back().emplace_back(*config.program, "network_layer_bottomright",
                                                &err);
    #ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  " Nw ->    " << op_flags << "," << network_forward_flags <<  std::endl;
    #endif
                    ASSERT_CL(err);
                    err = kernels.back().back().setArg(0, op_flags);
                    ASSERT_CL(err)
                    err = kernels.back().back().setArg(1, network_forward_flags);
                    ASSERT_CL(err)

                    err = network_queues_bottomright.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err)   
                }
                // Create the network kernel
                kernels.back().emplace_back(*config.program, "network_layer_topleft",
                                            &err);
    #ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  " Nw <-    " << op_flags << "," << network_forward_flags <<  std::endl;
    #endif
                ASSERT_CL(err);
                err = kernels.back().back().setArg(0, op_flags);
                ASSERT_CL(err)
                err = kernels.back().back().setArg(1, network_forward_flags);
                ASSERT_CL(err)

                err = network_queues_topleft.back().enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err)  


#ifndef NDEBUG
                std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col <<  " IS " << op_flags  <<  std::endl;
#endif

                kernels.back().emplace_back(*config.program, "inner_store",
                            &err);
                err = kernels.back().back().setArg(0, (left_block_is_received) ? left_buffers.back().back() : Buffer_network_scaling);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(1, (top_block_is_received) ? top_buffers.back().back() : Buffer_network_scaling);
                ASSERT_CL(err);
                err = kernels.back().back().setArg(2, op_flags);
                ASSERT_CL(err);

                if (std::distance(it,network_layer_op_flags.end()) == 1) {
                    all_events.back().emplace_back();
                    err = inner_queues.back()[0].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                    ASSERT_CL(err) 
                }
                else {
                    err = inner_queues.back()[0].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err)    
                }

                nw_exe_count++;

            }

            // update all remaining inner blocks using only global memory

            // only emplace new event list, if the inner mm kernel will be executed
            // otherwise the runtime dependency between the kernels may get lost!
            all_events.emplace_back(all_events.back());
            uint current_update = 0;
            uint current_replication = 0;
//             uint total_inner_updates_first_row = top_buffers.back().size();
//             uint updates_per_replication = total_inner_updates_first_row / config.programSettings->kernelReplications;
//             for (auto l = std::next(left_buffers.back().begin()); l < left_buffers.back().end(); l++) {
//                 // select the matrix multiplication kernel that should be used for this block updated 
//                 kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
//                                     &err);

//                 int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols);
//                 int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows + std::distance(left_buffers.back().begin(), l));
// #ifndef NDEBUG
//                 std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
// #endif   
//                 ASSERT_CL(err);
//                 err = kernels.back().back().setArg(0, Buffer_a);
//                 ASSERT_CL(err);
//                 err = kernels.back().back().setArg(1, *l);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(2, *top_buffers.back().begin());
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(3, block_col);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(4, block_row);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(5, blocks_per_row);
//                 ASSERT_CL(err)

//                 // Distribute the workload over all available matrix multiplication kernels
//                 err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
//                 current_replication = (current_replication + 1) % config.programSettings->kernelReplications;
//                 ASSERT_CL(err) 
//             }
//             current_replication = 0;
//             for (auto t = top_buffers.back().begin(); t < top_buffers.back().end(); t++) {
//                 // select the matrix multiplication kernel that should be used for this block updated 
//                 kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
//                                     &err);

//                 int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols + std::distance(top_buffers.back().begin(), t));
//                 int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows);
// #ifndef NDEBUG
//                 std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
// #endif   
//                 ASSERT_CL(err);
//                 err = kernels.back().back().setArg(0, Buffer_a);
//                 ASSERT_CL(err);
//                 err = kernels.back().back().setArg(1, *left_buffers.back().begin());
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(2, *t);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(3, block_col);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(4, block_row);
//                 ASSERT_CL(err)
//                 err = kernels.back().back().setArg(5, blocks_per_row);
//                 ASSERT_CL(err)
//                 // If number of blocks is not dividable by the number of replications, the first replications will do one update more
//                 uint updates_for_current_replication = updates_per_replication + ((current_replication < total_inner_updates_first_row % config.programSettings->kernelReplications) ? 1 : 0);
//                 if ((current_update + 1) == updates_for_current_replication) {
//                     // this is the last taks that will be enqueued in this queue, so create an event
//                     all_events.back().emplace_back();
//                     // Distribute the workload over all available matrix multiplication kernels
//                     err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));         
//                     current_update = 0;
//                     current_replication++;
//                 }
//                 else {
//                     // Distribute the workload over all available matrix multiplication kernels
//                     err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
//                     current_update++;
//                 }
//                 ASSERT_CL(err) 
//             }
//             current_replication = 0;
//             for (auto l = std::next(left_buffers.back().begin()); l < left_buffers.back().end(); l++) {
//                 for (auto t = std::next(top_buffers.back().begin()); t < top_buffers.back().end(); t++) {
//                     // select the matrix multiplication kernel that should be used for this block updated 
//                     kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
//                                         &err);

//                     int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols + std::distance(top_buffers.back().begin(), t));
//                     int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows + std::distance(left_buffers.back().begin(), l));
// #ifndef NDEBUG
//                     std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
// #endif   
//                     ASSERT_CL(err);
//                     err = kernels.back().back().setArg(0, Buffer_a);
//                     ASSERT_CL(err);
//                     err = kernels.back().back().setArg(1, *l);
//                     ASSERT_CL(err)
//                     err = kernels.back().back().setArg(2, *t);
//                     ASSERT_CL(err)
//                     err = kernels.back().back().setArg(3, block_col);
//                     ASSERT_CL(err)
//                     err = kernels.back().back().setArg(4, block_row);
//                     ASSERT_CL(err)
//                     err = kernels.back().back().setArg(5, blocks_per_row);
//                     ASSERT_CL(err)

//                     // Distribute the workload over all available matrix multiplication kernels
//                     err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
//                     current_replication = (current_replication + 1) % config.programSettings->kernelReplications;

//                     ASSERT_CL(err)
            uint total_inner_updates = left_buffers.back().size() * top_buffers.back().size();
            uint updates_per_replication = total_inner_updates / config.programSettings->kernelReplications;
            for (auto l = left_buffers.back().begin(); l < left_buffers.back().end(); l++) {
                for (auto t = top_buffers.back().begin(); t < top_buffers.back().end(); t++) {
                    // select the matrix multiplication kernel that should be used for this block updated 
                    kernels.back().emplace_back(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                        &err);

                    int block_col = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_cols + std::distance(top_buffers.back().begin(), t));
                    int block_row = static_cast<cl_uint>((config.programSettings->matrixSize / config.programSettings->blockSize) - num_inner_block_rows + std::distance(left_buffers.back().begin(), l));
#ifndef NDEBUG
                    std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Inner " << block_row << "," << block_col <<  std::endl;
#endif   
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
                    uint updates_for_current_replication = updates_per_replication + ((current_replication < total_inner_updates % config.programSettings->kernelReplications) ? 1 : 0);
                    if ((current_update + 1) == updates_for_current_replication) {
                        // this is the last taks that will be enqueued in this queue, so create an event
                        all_events.back().emplace_back();
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));         
                        current_update = 0;
                        current_replication++;
                    }
                    else {
                        // Distribute the workload over all available matrix multiplication kernels
                        err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(kernels.back().back(), cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
                        current_update++;
                    }
                }
            }
#ifndef NDEBUG
            MPI_Barrier(MPI_COMM_WORLD);
            if (is_calulating_lu_block) std::cout << "---------------" << std::endl;
#endif
            // Execute GEFA
            if (block_row == 0) {
                t1 = std::chrono::high_resolution_clock::now();
                // Trigger the user event that will start the first tasks in the queue
                start_event.setStatus(CL_COMPLETE);
            }

            if (block_row == blocks_per_row - 1) {
                // wait until the last LU queue is done since it will be the last required operation
                lu_queues.back().finish();
                t2 = std::chrono::high_resolution_clock::now();

                // Finish all other queues
                lu_queues.back().finish();
                network_queues_bottomright.back().finish();
                network_queues_topleft.back().finish();
                top_queues.back().finish();
                left_queues.back().finish();

            }
            
#ifndef NDEBUG
            network_queues_bottomright.back().finish();
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " NW -> Done    " << block_row <<  std::endl;
            network_queues_topleft.back().finish();
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " NW <- Done    " << block_row <<  std::endl;
            inner_queues.back()[0].finish();
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " IS Done    " << block_row <<  std::endl;
            cl::Event::waitForEvents(all_events.back());
            std::cout << "Torus " << config.programSettings->torus_row << "," << config.programSettings->torus_col << " Done    " << block_row <<  std::endl;
#endif
            if (block_row > 1) {
                if (block_row == 2) {
                    // additionally remove the user event in the first cleanup
                    all_events.pop_front();
                }

                // For the MM, an additional list of events was created. Check if it is already done
                // and pop it afterwards
                twait1 = std::chrono::high_resolution_clock::now();
                cl::Event::waitForEvents(*std::next(all_events.begin()));
                twait2 = std::chrono::high_resolution_clock::now();
                currentwaittime += std::chrono::duration_cast<std::chrono::duration<double>>
                                                                    (twait2 - twait1);
                all_events.pop_front();
                all_events.pop_front();

                lu_queues.pop_front();
                network_queues_bottomright.pop_front();
                network_queues_topleft.pop_front();
                top_queues.pop_front();
                left_queues.pop_front();

                left_buffers.pop_front();
                top_buffers.pop_front();
                // remove inner block queues 
                // (now MM events are all completed!)
                inner_queues.pop_front();
                kernels.pop_front();
            }


        }

        std::cout << "Wait time: " << currentwaittime.count() << "s" << std::endl;

#ifndef NDEBUG
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

    std::unique_ptr<linpack::LinpackExecutionTimings> results(
                    new linpack::LinpackExecutionTimings{gefaExecutionTimes, geslExecutionTimes});

    return results;
}

}  // namespace bm_execution

