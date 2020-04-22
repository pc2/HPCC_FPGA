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
#include "src/host/execution.h"

/* C++ standard library headers */
#include <chrono>
#include <fstream>
#include <memory>
#include <vector>

#ifdef DEBUG
#include <iostream>
#endif

/* External library headers */
#include "CL/cl.hpp"
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif

/* Project's headers */
#include "src/host/fpga_setup.h"
#include "src/host/linpack_functionality.h"

namespace bm_execution {

/*
 Prepare kernels and execute benchmark for the blocked approach

 @copydoc bm_execution::calculate()
*/
std::shared_ptr<ExecutionResults>
calculate(cl::Context context, cl::Device device, cl::Program program,
          uint repetitions, ulong matrixSize, uint blockSize) {
    uint lda = matrixSize;
    DATA_TYPE* a;
    posix_memalign(reinterpret_cast<void**>(&a), 64,
                  sizeof(DATA_TYPE)*lda*matrixSize);
    DATA_TYPE* b;
    posix_memalign(reinterpret_cast<void**>(&b), 64,
                  sizeof(DATA_TYPE)* matrixSize);
    cl_int* ipvt;
    posix_memalign(reinterpret_cast<void**>(&ipvt), 64,
                  sizeof(cl_int) * matrixSize);

    for (int i = 0; i < matrixSize; i++) {
        ipvt[i] = i;
    }

    DATA_TYPE norma = 0;
    double ops = (2.0e0*(matrixSize*matrixSize*matrixSize))/
                 3.0 + 2.0*(matrixSize*matrixSize);
    int err;

    // Create Command queue
    cl::CommandQueue compute_queue(context, device);

    // Create Buffers for input and output
    cl::Buffer Buffer_a(context, CL_MEM_READ_WRITE,
                                        sizeof(DATA_TYPE)*lda*matrixSize);

    // create the kernels
    cl::Kernel gefakernel(program, GEFA_KERNEL,
                                    &err);
    ASSERT_CL(err);


    // prepare kernels
    err = gefakernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = gefakernel.setArg(1, static_cast<uint>(matrixSize / blockSize));
    ASSERT_CL(err);

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> executionTimes;
    for (int i = 0; i < repetitions; i++) {
        matgen(a, lda, matrixSize, b, &norma);
        compute_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(DATA_TYPE)*lda*matrixSize, a);
        compute_queue.finish();
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

    compute_queue.enqueueReadBuffer(Buffer_a, CL_TRUE, 0,
                                     sizeof(DATA_TYPE)*lda*matrixSize, a);

#ifdef DEBUG
    for (int i= 0; i < matrixSize; i++) {
        for (int j=0; j < matrixSize; j++) {
            std::cout << a[i*lda + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout <<  std::endl;
#endif

    gesl_ref(a, b, ipvt, matrixSize, matrixSize);

    /* --- Check Results --- */

    double error = checkLINPACKresults(b, matrixSize, matrixSize);

    /* Check CPU reference results */

    matgen(a, lda, matrixSize, b, &norma);
    gefa_ref(a, matrixSize, lda, ipvt);

#ifdef DEBUG
    for (int i= 0; i < matrixSize; i++) {
        for (int j=0; j < matrixSize; j++) {
            std::cout << a[i*lda + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout <<  std::endl;
#endif

    gesl_ref(a, b, ipvt, matrixSize, matrixSize);
    checkLINPACKresults(b, matrixSize, matrixSize);

    free(reinterpret_cast<void *>(a));
    free(reinterpret_cast<void *>(b));
    free(reinterpret_cast<void *>(ipvt));

    std::shared_ptr<ExecutionResults> results(
                    new ExecutionResults{executionTimes,
                                         error});
    return results;
}

}  // namespace bm_execution
