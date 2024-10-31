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
#ifdef INTEL_FPGA
#include "CL/cl_ext_intelfpga.h"
#endif

namespace bm_execution {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    std::map<std::string, std::vector<double>>
    calculate(hpcc_base::ExecutionSettings<random_access::RandomAccessProgramSettings, cl::Device, cl::Context, cl::Program> const& config, HOST_DATA_TYPE * data, int mpi_rank, int mpi_size) {
        // int used to check for OpenCL errors
        int err;

        std::vector<cl::CommandQueue> compute_queue;
        std::vector<cl::Buffer> Buffer_data;
        std::vector<cl::Buffer> Buffer_randoms;
        std::vector<cl::Kernel> accesskernel;

        // Calculate RNG initial values
        HOST_DATA_TYPE* random_inits;
        posix_memalign(reinterpret_cast<void**>(&random_inits), 4096, sizeof(HOST_DATA_TYPE)*config.programSettings->numRngs);
        HOST_DATA_TYPE chunk = config.programSettings->dataSize * mpi_size * 4 / std::min(static_cast<size_t>(config.programSettings->numRngs), config.programSettings->dataSize * 4 * mpi_size);
        HOST_DATA_TYPE ran = 1;
        random_inits[0] = ran;
        for (HOST_DATA_TYPE r=0; r < config.programSettings->numRngs - 1; r++) {
            for (HOST_DATA_TYPE run = 0; run < chunk; run++) {
                HOST_DATA_TYPE_SIGNED v = 0;
                if (((HOST_DATA_TYPE_SIGNED) ran) < 0) {
                    v = POLY;
                }
                ran = (ran << 1) ^v;
            }
            random_inits[r + 1] = ran;
        }


        /* --- Prepare kernels --- */

        for (int r=0; r < config.programSettings->kernelReplications; r++) {
            compute_queue.push_back(cl::CommandQueue(*config.context, *config.device, 0, &err));
            ASSERT_CL(err);
            int memory_bank_info = 0;
#ifdef INTEL_FPGA
#ifdef USE_HBM
            memory_bank_info = CL_MEM_HETEROGENEOUS_INTELFPGA;
#else
            memory_bank_info = ((r + 1) << 16);
#endif
#endif
            Buffer_data.push_back(cl::Buffer(*config.context,
                        CL_MEM_READ_WRITE | memory_bank_info,
                        sizeof(HOST_DATA_TYPE)*(config.programSettings->dataSize / config.programSettings->kernelReplications)));

            Buffer_randoms.emplace_back(*config.context,
                        CL_MEM_READ_ONLY,
                        sizeof(HOST_DATA_TYPE)*config.programSettings->numRngs);
#ifdef INTEL_FPGA
            accesskernel.push_back(cl::Kernel(*config.program,
                        (RANDOM_ACCESS_KERNEL + std::to_string(r)).c_str() ,
                        &err));
#endif
#ifdef XILINX_FPGA
            accesskernel.push_back(cl::Kernel(*config.program,
                        (std::string(RANDOM_ACCESS_KERNEL) + "0:{" + RANDOM_ACCESS_KERNEL + "0_" + std::to_string(r + 1) + "}").c_str() ,
                        &err));
#endif
           ASSERT_CL(err);

            // prepare kernels
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(accesskernel[r](), 0,
                                        reinterpret_cast<void*>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]));
            err = clSetKernelArgSVMPointer(accesskernel[r](), 1,
                                        reinterpret_cast<void*>(random_inits));
#else
            err = accesskernel[r].setArg(0, Buffer_data[r]);
#endif

            ASSERT_CL(err);
            err = accesskernel[r].setArg(1, Buffer_randoms[r]);
            ASSERT_CL(err);
            err = accesskernel[r].setArg(2, HOST_DATA_TYPE(config.programSettings->dataSize * mpi_size));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(3,
                                         HOST_DATA_TYPE((config.programSettings->dataSize / config.programSettings->kernelReplications)));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(4,(1));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(5,
                                         cl_uint(mpi_rank * config.programSettings->kernelReplications + r));
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
                    err = clEnqueueSVMMap(compute_queue[r](), CL_TRUE,
                                    CL_MAP_READ | CL_MAP_WRITE,
                                    reinterpret_cast<void *>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]),
                                    sizeof(HOST_DATA_TYPE) *
                                    (config.programSettings->dataSize / config.programSettings->kernelReplications), 0,
                                    NULL, NULL);
                    ASSERT_CL(err)
                    err = clEnqueueSVMMap(compute_queue[r](), CL_TRUE,
                                    CL_MAP_READ,
                                    reinterpret_cast<void *>(random_inits),
                                    sizeof(HOST_DATA_TYPE)  * config.programSettings->numRngs, 0,
                                    NULL, NULL);
                    ASSERT_CL(err)
#else
                    err = compute_queue[r].enqueueWriteBuffer(Buffer_data[r], CL_TRUE, 0,
                                                        sizeof(HOST_DATA_TYPE) *
                                                        (config.programSettings->dataSize / config.programSettings->kernelReplications),
                                                        &data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]);
                    ASSERT_CL(err)
                    err = compute_queue[r].enqueueWriteBuffer(Buffer_randoms[r], CL_TRUE, 0,
                                                        sizeof(HOST_DATA_TYPE) * config.programSettings->numRngs,
                                                        random_inits);
                    ASSERT_CL(err)
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
                    compute_queue[r].enqueueNDRangeKernel(accesskernel[r], cl::NullRange, cl::NDRange(1));
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
            err = clEnqueueSVMUnmap(compute_queue[r](),
                                reinterpret_cast<void *>(&data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]), 0,
                                NULL, NULL);
            err = clEnqueueSVMUnmap(compute_queue[r](),
                                reinterpret_cast<void *>(random_inits), 0,
                                NULL, NULL);
#else
            err = compute_queue[r].enqueueReadBuffer(Buffer_data[r], CL_TRUE, 0,
                    sizeof(HOST_DATA_TYPE)*(config.programSettings->dataSize / config.programSettings->kernelReplications), 
                    &data[r * (config.programSettings->dataSize / config.programSettings->kernelReplications)]);
#endif
            ASSERT_CL(err)
        }

        free(random_inits);

        std::map<std::string, std::vector<double>> timings;

        timings["execution"] = executionTimes;

        return timings;
    }
}  // namespace bm_execution
