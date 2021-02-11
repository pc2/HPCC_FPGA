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

    uint blocks_per_row = config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG;

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
                                        sizeof(HOST_DATA_TYPE)*(1 << LOCAL_MEM_BLOCK_LOG)*(1 << LOCAL_MEM_BLOCK_LOG));
    cl::Buffer Buffer_lu2(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(1 << LOCAL_MEM_BLOCK_LOG)*(1 << LOCAL_MEM_BLOCK_LOG));
    cl::Buffer Buffer_top(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize * (1 << LOCAL_MEM_BLOCK_LOG));
    cl::Buffer Buffer_left(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize * (1 << LOCAL_MEM_BLOCK_LOG));
    cl::Buffer Buffer_network_scaling(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*(1 << LOCAL_MEM_BLOCK_LOG));

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> gefaExecutionTimes;
    std::vector<double> geslExecutionTimes;
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
        std::list<cl::CommandQueue> network_queues;
        std::list<std::vector<cl::CommandQueue>> inner_queues;
        std::list<bool> mm_was_used;

        // User event that is used to start actual execution of benchmark kernels
        cl::UserEvent start_event(*config.context, &err);
        ASSERT_CL(err);
        std::list<std::vector<cl::Event>> all_events;
        all_events.emplace_back();
        all_events.back().emplace_back(start_event);

        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;


        // For every row of blocks create kernels and enqueue them
        for (int block_row=0; block_row < config.programSettings->matrixSize / config.programSettings->blockSize; block_row++) {

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
            network_queues.emplace_back(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)

            all_events.emplace_back();
            // create the LU kernel
            cl::Kernel gefakernel(*config.program, "lu",
                                        &err);
            err = gefakernel.setArg(0, Buffer_a);
            ASSERT_CL(err);
            err = gefakernel.setArg(1, block_row);
            ASSERT_CL(err)
            err = gefakernel.setArg(2, block_row);
            ASSERT_CL(err)
            err = gefakernel.setArg(3, config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG);
            ASSERT_CL(err)
            all_events.back().emplace_back();
            err = lu_queues.back().enqueueNDRangeKernel(gefakernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &all_events.back().back());
            ASSERT_CL(err)
            // Create top kernels, left kernels and inner kernels
            for (int tops=block_row + 1; tops < (config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG); tops++) {
                cl::Kernel topkernel(*config.program, "top_update",
                                                &err);
                ASSERT_CL(err);     
                err = topkernel.setArg(0, Buffer_a);
                ASSERT_CL(err);    
                err = topkernel.setArg(1, Buffer_lu1);
                ASSERT_CL(err) 
                err = topkernel.setArg(2, (tops == block_row + 1) ? CL_TRUE : CL_FALSE);
                ASSERT_CL(err) 
                err = topkernel.setArg(3, tops);
                ASSERT_CL(err)
                err = topkernel.setArg(4, block_row);
                ASSERT_CL(err)
                err = topkernel.setArg(5, config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG);
                ASSERT_CL(err)
                err = top_queues.back().enqueueNDRangeKernel(topkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err)  

                cl::Kernel leftkernel(*config.program, "left_update",
                                                &err);
                ASSERT_CL(err);     
                err = leftkernel.setArg(0, Buffer_a);
                ASSERT_CL(err);    
                err = leftkernel.setArg(1, Buffer_lu2);
                ASSERT_CL(err) 
                err = leftkernel.setArg(2, (tops == block_row + 1) ? CL_TRUE : CL_FALSE);
                ASSERT_CL(err) 
                err = leftkernel.setArg(3, block_row);
                ASSERT_CL(err)
                err = leftkernel.setArg(4, tops);
                ASSERT_CL(err)
                err = leftkernel.setArg(5, config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG);
                ASSERT_CL(err)
                err = left_queues.back().enqueueNDRangeKernel(leftkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err) 

                // Create the network kernel
                cl::Kernel networkkernel(*config.program, "network_layer",
                                            &err);
                ASSERT_CL(err);
                err = networkkernel.setArg(0, Buffer_network_scaling);
                ASSERT_CL(err);
                err = networkkernel.setArg(1, TOP_BLOCK_OUT | LEFT_BLOCK_OUT | INNER_BLOCK | ((tops == block_row + 1) ? (LU_BLOCK_OUT | TOP_BLOCK | LEFT_BLOCK) : 0));
                ASSERT_CL(err)
                err = networkkernel.setArg(2, static_cast<cl_uint>(0));
                ASSERT_CL(err)
                err = network_queues.back().enqueueNDRangeKernel(networkkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err) 

                // create all diagnonal inner updates because we need to receive the data from top and left while calculating
                cl::Kernel innerkernel(*config.program, "inner_update",
                                    &err);
                ASSERT_CL(err);
                err = innerkernel.setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = innerkernel.setArg(1, Buffer_left);
                ASSERT_CL(err)
                err = innerkernel.setArg(2, Buffer_top);
                ASSERT_CL(err)
                // TODO remove this legacy code
                // err = innerkernel.setArg(3, CL_TRUE);
                // ASSERT_CL(err)
                // err = innerkernel.setArg(4, tops);
                // ASSERT_CL(err)
                // err = innerkernel.setArg(5, tops);
                // ASSERT_CL(err)
                // err = innerkernel.setArg(6, blocks_per_row);
                // ASSERT_CL(err)
                err = innerkernel.setArg(3, tops);
                ASSERT_CL(err)
                err = innerkernel.setArg(4, tops);
                ASSERT_CL(err)
                err = innerkernel.setArg(5, blocks_per_row);
                ASSERT_CL(err)
                if (tops + 1 ==  (config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG)) {
                    // only create an event for the last kernel execution that is enqueued
                    all_events.back().emplace_back();
                    err = inner_queues.back()[0].enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));
                    ASSERT_CL(err) 
                }
                else {
                    err = inner_queues.back()[0].enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                    ASSERT_CL(err) 
                }

            }
            // update all remaining inner blocks using only global memory
            uint total_inner_updates = (blocks_per_row - (block_row + 1) - 1) * (blocks_per_row - (block_row + 1));
            uint updates_per_replication = total_inner_updates / config.programSettings->kernelReplications;
            mm_was_used.push_back(total_inner_updates > 0);
            if (total_inner_updates > 0) {
                // only emplace new event list, if the inner mm kernel will be executed
                // otherwise the runtime dependency between the kernels may get lost!
                all_events.emplace_back();
                uint current_update = 0;
                uint current_replication = 0;
                for (int current_row=block_row + 1; current_row < blocks_per_row; current_row++) {
                    for (int current_col=block_row + 1; current_col < blocks_per_row; current_col++) {
                        if (current_row == current_col) {
                            continue;
                        }
                        // select the matrix multiplication kernel that should be used for this block updated 
                        // with a round-robin scheme
                        cl::Kernel innerkernel(*config.program, ("inner_update_mm" + std::to_string(current_replication)).c_str(),
                                            &err);
                        ASSERT_CL(err);
                        err = innerkernel.setArg(0, Buffer_a);
                        ASSERT_CL(err);
                        err = innerkernel.setArg(1, Buffer_left);
                        ASSERT_CL(err)
                        err = innerkernel.setArg(2, Buffer_top);
                        ASSERT_CL(err)
                        err = innerkernel.setArg(3, current_col);
                        ASSERT_CL(err)
                        err = innerkernel.setArg(4, current_row);
                        ASSERT_CL(err)
                        err = innerkernel.setArg(5, blocks_per_row);
                        ASSERT_CL(err)
                        // If number of blocks is not dividable by the number of replications, the first replications will do one update more
                        uint updates_for_current_replication = updates_per_replication + ((current_replication < total_inner_updates % config.programSettings->kernelReplications) ? 1 : 0);
                        if ((current_update + 1) == updates_for_current_replication) {
                            // this is the last taks that will be enqueued in this queue, so create an event
                            all_events.back().emplace_back();
                            // Distribute the workload over all available matrix multiplication kernels
                            err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))), &(all_events.back().back()));         
                            current_update = 0;
                            current_replication++;
                        }
                        else {
                            // Distribute the workload over all available matrix multiplication kernels
                            err = inner_queues.back()[(current_replication) + 1].enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));         
                            current_update++;
                        }
                        ASSERT_CL(err) 
                    }
                }
            }

            if (block_row == blocks_per_row - 1) {
                // Create the network kernel for the very last iteration where only the LU kernel will run
                cl::Kernel networkkernel(*config.program, "network_layer",
                                            &err);
                ASSERT_CL(err);
                err = networkkernel.setArg(0, Buffer_network_scaling);
                ASSERT_CL(err);
                err = networkkernel.setArg(1, LU_BLOCK_OUT);
                ASSERT_CL(err)
                err = networkkernel.setArg(2, static_cast<cl_uint>(0));
                ASSERT_CL(err)
                err = network_queues.back().enqueueNDRangeKernel(networkkernel, cl::NullRange, cl::NDRange(1), cl::NullRange,  &(*std::prev(std::prev(all_events.end()))));
                ASSERT_CL(err) 
            }

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
            }
            if (block_row > 1) {
                if (block_row == 2) {
                    // additionally remove the user event in the first cleanup
                    all_events.pop_front();
                }
                // Wait for all events that where executed 
                // two iterations before and remove them from the data structure
                // to free resources
                cl::Event::waitForEvents(all_events.front());
                all_events.pop_front();
                lu_queues.pop_front();
                network_queues.pop_front();
                top_queues.pop_front();
                left_queues.pop_front();
                if (mm_was_used.front()) {
                    // For the MM, an additional list of events was created. Check if it is already done
                    // and pop it afterwards
                    cl::Event::waitForEvents(all_events.front());
                    all_events.pop_front();
                }
                // remove inner block queues 
                // (now MM events are all completed!)
                inner_queues.pop_front();
                // pop flag that indicates if MM was used in the iteration that is handeled now
                mm_was_used.pop_front();
            }


        }

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
    buffer_queue.enqueueReadBuffer(Buffer_b, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
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
