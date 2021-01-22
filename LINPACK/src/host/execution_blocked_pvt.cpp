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
#include "CL/cl.hpp"
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

    // Create Buffers for input and output
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_b(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize);
    cl::Buffer Buffer_pivot(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int)*config.programSettings->matrixSize);

    // create the kernels
    cl::Kernel gefakernel(*config.program, "lu",
                                    &err);
    cl::Kernel gefa2kernel(*config.program, "lu",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel topkernel(*config.program, "top_update",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel leftkernel(*config.program, "left_update",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel innerkernel(*config.program, "inner_update",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel geslkernel(*config.program, "gesl",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel network1kernel(*config.program, "network_layer",
                                    &err);
    ASSERT_CL(err);
    cl::Kernel network2kernel(*config.program, "network_layer",
                                    &err);
    ASSERT_CL(err);

    err = network1kernel.setArg(0, TOP_BLOCK | LEFT_BLOCK | INNER_BLOCK | LU_BLOCK);
    err = network1kernel.setArg(1, CL_FALSE);
    err = network2kernel.setArg(0, LU_BLOCK);
    err = network2kernel.setArg(1, CL_FALSE);

    // prepare kernels
#ifdef USE_SVM
    // To prevent the reuse of the result of previous repetitions, use this
    // buffer instead and copy the result back to the real buffer 
    HOST_DATA_TYPE* A_tmp = reinterpret_cast<HOST_DATA_TYPE*>(
                    clSVMAlloc((*config.context)(), 0 ,
                    config.programSettings->matrixSize * 
                    config.programSettings->matrixSize * sizeof(HOST_DATA_TYPE), 1024));

    err = clSetKernelArgSVMPointer(gefakernel(), 0,
                                    reinterpret_cast<void*>(A_tmp));
    ASSERT_CL(err)
    if (!config.programSettings->isDiagonallyDominant) {
        err = clSetKernelArgSVMPointer(gefakernel(), 1,
                                        reinterpret_cast<void*>(ipvt));
        ASSERT_CL(err)
    }
#else
    err = gefakernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = gefa2kernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = topkernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = leftkernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = innerkernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
#endif
    err = gefakernel.setArg(1, 0);
    err = gefa2kernel.setArg(1, 1);
    err = topkernel.setArg(1, 1);
    err = leftkernel.setArg(1, 0);
    err = innerkernel.setArg(1, 1);
    err = gefakernel.setArg(2, 0);
    err = gefa2kernel.setArg(2, 1);
    err = topkernel.setArg(2, 0);
    err = leftkernel.setArg(2, 1);
    err = innerkernel.setArg(2, 1);
    err = gefakernel.setArg(3, 2);
    err = gefa2kernel.setArg(3, 2);
    err = topkernel.setArg(3, 2);
    err = leftkernel.setArg(3, 2);
    err = innerkernel.setArg(3, 2);

#ifdef USE_SVM

    err = clSetKernelArgSVMPointer(geslkernel(), 0,
                                    reinterpret_cast<void*>(A_tmp));
    ASSERT_CL(err)
    err = clSetKernelArgSVMPointer(geslkernel(), 1,
                                    reinterpret_cast<void*>(b));
    ASSERT_CL(err)
    if (!config.programSettings->isDiagonallyDominant) {
        err = clSetKernelArgSVMPointer(geslkernel(), 2,
                                        reinterpret_cast<void*>(ipvt));
        ASSERT_CL(err)
    }
#else
    err = geslkernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = geslkernel.setArg(1, Buffer_b);
    ASSERT_CL(err);
    if (!config.programSettings->isDiagonallyDominant) {
        err = geslkernel.setArg(2, Buffer_pivot);
        ASSERT_CL(err);
    }
#endif
    err = geslkernel.setArg(config.programSettings->isDiagonallyDominant ? 2 : 3, static_cast<uint>(config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG));
    ASSERT_CL(err);



    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> gefaExecutionTimes;
    std::vector<double> geslExecutionTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {
#ifdef USE_SVM
        for (int k=0; k < config.programSettings->matrixSize * config.programSettings->matrixSize; k++) {
            A_tmp[k] = A[k];
        }

        err = clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_READ | CL_MAP_WRITE,
                        reinterpret_cast<void *>(A_tmp),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
        err = clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(b),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
        err = clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_WRITE,
                        reinterpret_cast<void *>(ipvt),
                        sizeof(cl_int) *
                        (config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
#else
        err = lu_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
        ASSERT_CL(err)
        err = lu_queue.enqueueWriteBuffer(Buffer_b, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
        ASSERT_CL(err)
        lu_queue.finish();
#endif
        std::vector<cl::Event> event_vector(5);
        // Execute GEFA
        auto t1 = std::chrono::high_resolution_clock::now();
        network_queue.enqueueTask(network1kernel, NULL, &event_vector[0]);
        lu_queue.enqueueTask(gefakernel, NULL, &event_vector[1]);
        top_queue.enqueueTask(topkernel, NULL, &event_vector[2]);
        left_queue.enqueueTask(leftkernel, NULL, &event_vector[3]);
        inner_queue.enqueueTask(innerkernel, NULL, &event_vector[4]);
        network_queue.enqueueTask(network2kernel, &event_vector, NULL);
        lu_queue.enqueueTask(gefa2kernel, &event_vector, NULL);
        lu_queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timespan =
            std::chrono::duration_cast<std::chrono::duration<double>>
                                                                (t2 - t1);
        gefaExecutionTimes.push_back(timespan.count());

        // Execute GESL
        t1 = std::chrono::high_resolution_clock::now();
        lu_queue.enqueueTask(geslkernel);
        lu_queue.finish();
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
    lu_queue.enqueueReadBuffer(Buffer_a, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
    lu_queue.enqueueReadBuffer(Buffer_b, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize, b);
    if (!config.programSettings->isDiagonallyDominant) {
        lu_queue.enqueueReadBuffer(Buffer_pivot, CL_TRUE, 0,
                                        sizeof(cl_int)*config.programSettings->matrixSize, ipvt);
    }
#endif

    std::unique_ptr<linpack::LinpackExecutionTimings> results(
                    new linpack::LinpackExecutionTimings{gefaExecutionTimes, geslExecutionTimes});
    return results;
}

}  // namespace bm_execution
