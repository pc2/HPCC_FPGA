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
    cl::CommandQueue compute_queue(*config.context, *config.device);

    // Create Buffers for input and output
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_pivot(*config.context, CL_MEM_READ_WRITE,
                                        sizeof(cl_int)*config.programSettings->matrixSize);

    // create the kernels
    cl::Kernel gefakernel(*config.program, "gefa",
                                    &err);
    ASSERT_CL(err);


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
    err = clSetKernelArgSVMPointer(gefakernel(), 1,
                                    reinterpret_cast<void*>(ipvt));
#else
    err = gefakernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = gefakernel.setArg(1, Buffer_pivot);
    ASSERT_CL(err);
#endif
    err = gefakernel.setArg(2, static_cast<uint>(config.programSettings->matrixSize >> LOCAL_MEM_BLOCK_LOG));
    ASSERT_CL(err);

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> executionTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {
#ifdef USE_SVM
        for (int k=0; k < config.programSettings->matrixSize * config.programSettings->matrixSize; k++) {
            A_tmp[k] = A[k];
        }

        clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_READ | CL_MAP_WRITE,
                        reinterpret_cast<void *>(A_tmp),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(b),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize), 0,
                        NULL, NULL);
        clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_WRITE,
                        reinterpret_cast<void *>(ipvt),
                        sizeof(cl_int) *
                        (config.programSettings->matrixSize), 0,
                        NULL, NULL);
#else
        compute_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
        compute_queue.finish();
#endif
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_queue.enqueueTask(gefakernel);
        compute_queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timespan =
            std::chrono::duration_cast<std::chrono::duration<double>>
                                                                (t2 - t1);
        executionTimes.push_back(timespan.count());
    }

    /* --- Read back results from Device --- */

#ifdef USE_SVM
    clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(A), 0,
                        NULL, NULL);
    clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(b), 0,
                        NULL, NULL);
    clEnqueueSVMUnmap(compute_queue(),
                        reinterpret_cast<void *>(ipvt), 0,
                        NULL, NULL);
    
    // read back result from temporary buffer
    for (int k=0; k < config.programSettings->matrixSize * config.programSettings->matrixSize; k++) {
        A[k] = A_tmp[k];
    }
    clSVMFree((*config.context)(), reinterpret_cast<void*>(A_tmp));

#else
    compute_queue.enqueueReadBuffer(Buffer_a, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, A);
    compute_queue.enqueueReadBuffer(Buffer_pivot, CL_TRUE, 0,
                                     sizeof(cl_int)*config.programSettings->matrixSize, ipvt);
#endif

    // Solve linear equations on CPU
    // TODO: This has to be done on FPGA
    linpack::gesl_ref(A, b, ipvt, config.programSettings->matrixSize, config.programSettings->matrixSize);

    std::unique_ptr<linpack::LinpackExecutionTimings> results(
                    new linpack::LinpackExecutionTimings{executionTimes});
    return results;
}

}  // namespace bm_execution
