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

    // Create Command queue
    cl::CommandQueue lu_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)
    cl::CommandQueue top_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)
    cl::CommandQueue left_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)
    cl::CommandQueue inner_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)
    cl::CommandQueue network_queue(*config.context, *config.device, 0, &err);
    ASSERT_CL(err)
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

    uint blocks_per_row = config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG;

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> gefaExecutionTimes;
    std::vector<double> geslExecutionTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {

        // User event that is used to start actual execution of benchmark kernels
        cl::UserEvent start_event(*config.context, &err);
        ASSERT_CL(err);
        std::vector<std::vector<cl::Event>> all_events(blocks_per_row + 1);
        all_events[0].push_back(start_event);

        // For every row of blocks create kernels and enqueue them
        for (int block_row=0; block_row < config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG; block_row++) {
            std::cout << "Create LU" << std::endl;
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
            all_events[block_row + 1].resize(all_events[block_row + 1].size() + 1);
            err = lu_queue.enqueueNDRangeKernel(gefakernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(all_events[block_row]), &all_events[block_row + 1][all_events[block_row + 1].size() - 1]);
            ASSERT_CL(err)
            // Create top kernels, left kernels and inner kernels
            for (int tops=block_row + 1; tops < (config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG); tops++) {
                std::cout << "Create top" << std::endl;
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
                all_events[block_row + 1].resize(all_events[block_row + 1].size() + 1);
                top_queue.enqueueNDRangeKernel(topkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(all_events[block_row]), &(all_events[block_row + 1][all_events[block_row + 1].size() - 1]));
                std::cout << "Create left" << std::endl;

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
                all_events[block_row + 1].resize(all_events[block_row + 1].size() + 1);
                left_queue.enqueueNDRangeKernel(leftkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(all_events[block_row]), &(all_events[block_row + 1][all_events[block_row + 1].size() - 1]));

                // Create the network kernel
                std::cout << "Create network" << std::endl;
                cl::Kernel networkkernel(*config.program, "network_layer",
                                            &err);
                ASSERT_CL(err);
                err = networkkernel.setArg(0, TOP_BLOCK_OUT | LEFT_BLOCK_OUT | INNER_BLOCK | ((tops == block_row + 1) ? (LU_BLOCK_OUT | TOP_BLOCK | LEFT_BLOCK) : 0));
                ASSERT_CL(err)
                err = networkkernel.setArg(1, CL_FALSE);
                ASSERT_CL(err)
                network_queue.enqueueNDRangeKernel(networkkernel, cl::NullRange, cl::NDRange(1), cl::NullRange);

                std::cout << "Create inner" << std::endl;
                cl::Kernel innerkernel(*config.program, "inner_update",
                                    &err);
                ASSERT_CL(err);
                err = innerkernel.setArg(0, Buffer_a);
                ASSERT_CL(err);
                err = innerkernel.setArg(1, Buffer_left);
                ASSERT_CL(err)
                err = innerkernel.setArg(2, Buffer_top);
                ASSERT_CL(err)
                err = innerkernel.setArg(3, CL_TRUE);
                ASSERT_CL(err)
                err = innerkernel.setArg(4, tops);
                ASSERT_CL(err)
                err = innerkernel.setArg(5, tops);
                ASSERT_CL(err)
                err = innerkernel.setArg(6, blocks_per_row);
                ASSERT_CL(err)
                all_events[block_row + 1].resize(all_events[block_row + 1].size() + 1);
                inner_queue.enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(all_events[block_row]), &(all_events[block_row + 1][all_events[block_row + 1].size() - 1]));
            }
            // remaining inner kernels
            for (int current_row=block_row + 1; current_row < blocks_per_row; current_row++) {
                for (int current_col=block_row + 1; current_col < blocks_per_row; current_col++) {
                    if (current_row == current_col) {
                        continue;
                    }
                    std::cout << "Create inner" << std::endl;
                    cl::Kernel innerkernel(*config.program, "inner_update",
                                        &err);
                    ASSERT_CL(err);
                    err = innerkernel.setArg(0, Buffer_a);
                    ASSERT_CL(err);
                    err = innerkernel.setArg(1, Buffer_left);
                    ASSERT_CL(err)
                    err = innerkernel.setArg(2, Buffer_top);
                    ASSERT_CL(err)
                    err = innerkernel.setArg(3, CL_FALSE);
                    ASSERT_CL(err)
                    err = innerkernel.setArg(4, current_col);
                    ASSERT_CL(err)
                    err = innerkernel.setArg(5, current_row);
                    ASSERT_CL(err)
                    err = innerkernel.setArg(6, blocks_per_row);
                    ASSERT_CL(err)
                    all_events[block_row + 1].resize(all_events[block_row + 1].size() + 1);
                    inner_queue.enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1), cl::NullRange, &(all_events[block_row]), &(all_events[block_row + 1][all_events[block_row + 1].size() - 1]));         
                }
            }

            if (block_row == blocks_per_row - 1) {
                std::cout << "Create network only LU" << std::endl;
                // Create the network kernel
                cl::Kernel networkkernel(*config.program, "network_layer",
                                            &err);
                ASSERT_CL(err);
                err = networkkernel.setArg(0, LU_BLOCK_OUT);
                ASSERT_CL(err)
                err = networkkernel.setArg(1, CL_FALSE);
                ASSERT_CL(err)
                network_queue.enqueueNDRangeKernel(networkkernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
            }
        }

        err = buffer_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
        ASSERT_CL(err)
        err = buffer_queue.enqueueWriteBuffer(Buffer_b, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
        ASSERT_CL(err)
        buffer_queue.finish();

        std::cout << "Start execution" << std::endl;
        // Execute GEFA
        auto t1 = std::chrono::high_resolution_clock::now();
        start_event.setStatus(CL_COMPLETE);
        std::cout << "Wait for iterations: " << all_events.size() << std::endl;
        for (auto evs : all_events) {
            std::cout << "Wait for events: " << evs.size() << std::flush;
            for (auto ev : evs) {
                ev.wait();
                std::cout << "." << std::flush;
            }
            std::cout << std::endl;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
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

    std::cout << "WARNING: GESL calculated on CPU!" << std::endl;
    linpack::gesl_ref_nopvt(A,b,config.programSettings->matrixSize,config.programSettings->matrixSize);

    std::unique_ptr<linpack::LinpackExecutionTimings> results(
                    new linpack::LinpackExecutionTimings{gefaExecutionTimes, geslExecutionTimes});

    return results;
}

}  // namespace bm_execution
