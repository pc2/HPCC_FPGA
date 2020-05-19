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
#ifdef INTEL_FPGA
#include "CL/cl_ext_intelfpga.h"
#endif


namespace bm_execution {

/*
 Prepare kernels and execute benchmark

 @copydoc bm_execution::calculate()
*/
std::unique_ptr<gemm::GEMMExecutionTimings>
calculate(hpcc_base::ExecutionSettings<gemm::GEMMProgramSettings> const& config, HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, HOST_DATA_TYPE* c, HOST_DATA_TYPE* c_out,
        HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta) {

    int err;

    // Create Command queue
    cl::CommandQueue compute_queue(*config.context, *config.device);
#ifdef INTEL_FPGA
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_1_INTELFPGA),
                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_b(*config.context, CL_MEM_READ_WRITE | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_2_INTELFPGA),
                        sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_c_in(*config.context, CL_MEM_READ_WRITE | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_3_INTELFPGA),
                           sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_c_out(*config.context, CL_MEM_READ_WRITE | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_4_INTELFPGA),
                            sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
#else
    cl::Buffer Buffer_a(*config.context, CL_MEM_READ_WRITE,
                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_b(*config.context, CL_MEM_READ_WRITE,
                        sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_c_in(*config.context, CL_MEM_READ_WRITE,
                           sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
    cl::Buffer Buffer_c_out(*config.context, CL_MEM_READ_WRITE,
                            sizeof(cl_int)*config.programSettings->matrixSize*config.programSettings->matrixSize);
#endif


    // create the kernels
    cl::Kernel gemmkernel(*config.program, KERNEL_NAME,
                                    &err);
    ASSERT_CL(err);


    // prepare kernels
    err = gemmkernel.setArg(0, Buffer_a);
    ASSERT_CL(err);
    err = gemmkernel.setArg(1, Buffer_b);
    ASSERT_CL(err);
    err = gemmkernel.setArg(2, Buffer_c_in);
    ASSERT_CL(err);
    err = gemmkernel.setArg(3, Buffer_c_out);
    ASSERT_CL(err);
    err = gemmkernel.setArg(4, alpha);
    ASSERT_CL(err);
    err = gemmkernel.setArg(5, beta);
    ASSERT_CL(err);
    err = gemmkernel.setArg(6, config.programSettings->matrixSize);
    ASSERT_CL(err);

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> executionTimes;
    for (int i = 0; i < config.programSettings->matrixSize; i++) {
        compute_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, a);
        compute_queue.enqueueWriteBuffer(Buffer_b, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, b);
        compute_queue.enqueueWriteBuffer(Buffer_c_in, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, c);
        compute_queue.finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_queue.enqueueTask(gemmkernel);
        compute_queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timespan =
            std::chrono::duration_cast<std::chrono::duration<double>>
                                                                (t2 - t1);
        executionTimes.push_back(timespan.count());
    }

    /* --- Read back results from Device --- */

    compute_queue.enqueueReadBuffer(Buffer_c_out, CL_TRUE, 0,
                                     sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, c_out);


    std::unique_ptr<gemm::GEMMExecutionTimings> results(
                    new gemm::GEMMExecutionTimings{executionTimes});
    return results;
}

}  // namespace bm_execution
