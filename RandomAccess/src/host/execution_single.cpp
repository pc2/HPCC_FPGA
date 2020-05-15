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


namespace bm_execution {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    std::shared_ptr<random_access::RandomAccessExecutionTimings>
    calculate(hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings> const& config, HOST_DATA_TYPE * data) {
        // int used to check for OpenCL errors
        int err;

        std::vector<cl::CommandQueue> compute_queue;
        std::vector<cl::Buffer> Buffer_data;
        std::vector<cl::Buffer> Buffer_random;
        std::vector<cl::Kernel> accesskernel;

        /* --- Prepare kernels --- */

        for (int r=0; r < config.programSettings->kernelReplications; r++) {
            compute_queue.push_back(cl::CommandQueue(config.context, config.device));

            int memory_bank_info = 0;
#ifdef INTEL_FPGA
            memory_bank_info = ((r + 1) << 16);
#endif
            Buffer_data.push_back(cl::Buffer(config.context,
                        CL_MEM_READ_WRITE | memory_bank_info,
                        sizeof(HOST_DATA_TYPE)*(config.programSettings->dataSize / config.programSettings->kernelReplications)));
#ifdef INTEL_FPGA
            accesskernel.push_back(cl::Kernel(config.program,
                        (RANDOM_ACCESS_KERNEL + std::to_string(r)).c_str() ,
                        &err));
#endif
#ifdef XILINX_FPGA
            accesskernel.push_back(cl::Kernel(config.program,
                        (std::string(RANDOM_ACCESS_KERNEL) + "0:{" + RANDOM_ACCESS_KERNEL + "0_" + std::to_string(r + 1) + "}").c_str() ,
                        &err));
#endif
           ASSERT_CL(err);

            // prepare kernels
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(accesskernel[r](), 0,
                                        reinterpret_cast<void*>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]));
#else
            err = accesskernel[r].setArg(0, Buffer_data[r]);
#endif

            ASSERT_CL(err);
            err = accesskernel[r].setArg(1, HOST_DATA_TYPE(config.programSettings->dataSize));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(2,
                                         HOST_DATA_TYPE((config.programSettings->dataSize / config.programSettings->kernelReplications)));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(3,
                                         cl_uint(r));
            ASSERT_CL(err);
        }

        /* --- Execute actual benchmark kernels --- */

        std::vector<double> executionTimes;
        for (int i = 0; i < config.programSettings->numRepetitions; i++) {
            std::chrono::time_point<std::chrono::high_resolution_clock> t1;
#pragma omp parallel default(shared)
            {
#pragma omp for
                for (int r = 0; r < config.programSettings->kernelReplications; r++) {
#ifdef USE_SVM
                    clEnqueueSVMMap(compute_queue[r](), CL_TRUE,
                                    CL_MAP_READ | CL_MAP_WRITE,
                                    reinterpret_cast<void *>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]),
                                    sizeof(HOST_DATA_TYPE) *
                                    (config.programSettings->dataSize / config.programSettings->kernelReplications), 0,
                                    NULL, NULL);
#else
                    compute_queue[r].enqueueWriteBuffer(Buffer_data[r], CL_TRUE, 0,
                                                        sizeof(HOST_DATA_TYPE) *
                                                        (config.programSettings->dataSize / config.programSettings->kernelReplications),
                                                        &data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]);
#endif
                }
#pragma omp master
                {
                    // Execute benchmark kernels
                    t1 = std::chrono::high_resolution_clock::now();
                }
#pragma omp barrier
#pragma omp for nowait
                for (int r = 0; r < config.programSettings->kernelReplications; r++) {
                    compute_queue[r].enqueueTask(accesskernel[r]);
                }
#pragma omp for
                for (int r = 0; r < config.programSettings->kernelReplications; r++) {
                    compute_queue[r].finish();
                }
#pragma omp master
                {
                    auto t2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> timespan =
                            std::chrono::duration_cast<std::chrono::duration<double>>
                                    (t2 - t1);
                    executionTimes.push_back(timespan.count());
                }
            }
        }

        /* --- Read back results from Device --- */
        for (int r=0; r < config.programSettings->kernelReplications; r++) {
#ifdef USE_SVM
            clEnqueueSVMUnmap(compute_queue[r](),
                                reinterpret_cast<void *>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]), 0,
                                NULL, NULL);
#else
            compute_queue[r].enqueueReadBuffer(Buffer_data[r], CL_TRUE, 0,
                    sizeof(HOST_DATA_TYPE)*(config.programSettings->dataSize / config.programSettings->kernelReplications), 
                    &data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]);
#endif
        }

        return std::shared_ptr<random_access::RandomAccessExecutionTimings>(new random_access::RandomAccessExecutionTimings{executionTimes});
    }

}  // namespace bm_execution
