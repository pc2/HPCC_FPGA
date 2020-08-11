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
    std::vector<cl::CommandQueue> compute_queues;
    for (int i=0; i < config.programSettings->kernelReplications; i++) {
        compute_queues.push_back(cl::CommandQueue(*config.context, *config.device, 0, &err));
        ASSERT_CL(err)
    }

    cl_int size_in_blocks = config.programSettings->matrixSize / config.programSettings->blockSize;
    size_t number_blocks_per_kernel = ((size_in_blocks + config.programSettings->kernelReplications - 1)/(config.programSettings->kernelReplications));
    size_t out_buffer_size = config.programSettings->matrixSize * 
                                (number_blocks_per_kernel) * config.programSettings->blockSize;

    std::vector<cl::Buffer> a_buffers;
    std::vector<cl::Buffer> b_buffers;
    std::vector<cl::Buffer> c_buffers;
    std::vector<cl::Buffer> out_buffers;

#ifdef INTEL_FPGA
    // Create an output buffer for every kernel to still allow restrict optimizations
    // For the other buffers this is not necessary, since they are read only
    for (int i=0; i < config.programSettings->kernelReplications; i++) {
        if (i == 0 || config.programSettings->replicateInputBuffers) {
            a_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_1_INTELFPGA),
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
            b_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_2_INTELFPGA),
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
            c_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_3_INTELFPGA),
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
        }
        out_buffers.push_back(cl::Buffer(*config.context, CL_MEM_WRITE_ONLY | (config.programSettings->useMemoryInterleaving ? 0 :CL_CHANNEL_4_INTELFPGA),
                                    sizeof(HOST_DATA_TYPE)*out_buffer_size, NULL, &err));
        ASSERT_CL(err)
    }
#else
    // Create an output buffer for every kernel to still allow restrict optimizations
    // For the other buffers this is not necessary, since they are read only
    for (int i=0; i < config.programSettings->kernelReplications; i++) {
        if (i == 0 || config.programSettings->replicateInputBuffers) {
            a_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY,
                                sizeof(HOST_DATA_TYPE) *config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
            b_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY,
                                sizeof(HOST_DATA_TYPE) *config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
            c_buffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_ONLY,
                                sizeof(HOST_DATA_TYPE) *config.programSettings->matrixSize*config.programSettings->matrixSize, NULL, &err));
            ASSERT_CL(err)
        }
        out_buffers.push_back(cl::Buffer(*config.context, CL_MEM_WRITE_ONLY,
                                    sizeof(HOST_DATA_TYPE) * out_buffer_size, NULL, &err));
        ASSERT_CL(err)
    }
#endif

    std::vector<cl::Kernel> gemmkernels;

    for (int i=0; i < config.programSettings->kernelReplications; i++) {
#ifdef INTEL_FPGA
        // create the kernels
        cl::Kernel gemmkernel(*config.program, (KERNEL_NAME + std::to_string(i)).c_str(),
                                        &err);
        ASSERT_CL(err);
#endif
#ifdef XILINX_FPGA
        // create the kernels
        cl::Kernel gemmkernel(*config.program, (std::string(KERNEL_NAME) + "0_" + std::to_string(r + 1) + ":{" + KERNEL_NAME + "0_" +  std::to_string(r + 1) + "}").c_str().c_str(),
                                        &err);
        ASSERT_CL(err);
#endif


        // prepare kernels
    #ifdef USE_SVM
        err = clSetKernelArgSVMPointer(gemmkernel(), 0,
                                        reinterpret_cast<void*>(a));
        ASSERT_CL(err)
        err = clSetKernelArgSVMPointer(gemmkernel(), 1,
                                        reinterpret_cast<void*>(b));
        ASSERT_CL(err)
        err = clSetKernelArgSVMPointer(gemmkernel(), 2,
                                        reinterpret_cast<void*>(c));
        ASSERT_CL(err)
        err = clSetKernelArgSVMPointer(gemmkernel(), 3,
                                        reinterpret_cast<void*>(&c_out[i * out_buffer_size]));
        ASSERT_CL(err)
    #else
        err = gemmkernel.setArg(0, a_buffers[config.programSettings->replicateInputBuffers ? i : 0]);
        ASSERT_CL(err);
        err = gemmkernel.setArg(1, b_buffers[config.programSettings->replicateInputBuffers ? i : 0]);
        ASSERT_CL(err);
        err = gemmkernel.setArg(2, c_buffers[config.programSettings->replicateInputBuffers ? i : 0]);
        ASSERT_CL(err);
        err = gemmkernel.setArg(3, out_buffers[i]);
        ASSERT_CL(err);
    #endif
        err = gemmkernel.setArg(4, alpha);
        ASSERT_CL(err);
        err = gemmkernel.setArg(5, beta);
        ASSERT_CL(err);
        err = gemmkernel.setArg(6, size_in_blocks);
        ASSERT_CL(err);
        err = gemmkernel.setArg(7, static_cast<cl_uint>(i * number_blocks_per_kernel));
        ASSERT_CL(err);
        err = gemmkernel.setArg(8, static_cast<cl_uint>(std::min<cl_uint>(i * number_blocks_per_kernel + number_blocks_per_kernel, size_in_blocks)));
        ASSERT_CL(err);

        gemmkernels.push_back(gemmkernel);
    }

    /* --- Execute actual benchmark kernels --- */

    double t;
    std::vector<double> executionTimes;
    for (int i = 0; i < config.programSettings->numRepetitions; i++) {
#ifdef USE_SVM
        err = clEnqueueSVMMap(compute_queues[0](), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(a),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
        err = clEnqueueSVMMap(compute_queues[0](), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(b),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
        err = clEnqueueSVMMap(compute_queues[0](), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(c),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
        err = clEnqueueSVMMap(compute_queues[0](), CL_TRUE,
                        CL_MAP_WRITE,
                        reinterpret_cast<void *>(c_out),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        ASSERT_CL(err)
#else

        for (int i=0; i < (config.programSettings->replicateInputBuffers ? config.programSettings->kernelReplications : 1); i++) {
            err = compute_queues[i].enqueueWriteBuffer(a_buffers[i], CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, a);
            ASSERT_CL(err)
            err = compute_queues[i].enqueueWriteBuffer(b_buffers[i], CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, b);
            ASSERT_CL(err)
            err = compute_queues[i].enqueueWriteBuffer(c_buffers[i], CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize, c);
            ASSERT_CL(err)
        }
        for (int i=0; i < config.programSettings->kernelReplications; i++) {
            compute_queues[i].finish();
        }
#endif
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i < config.programSettings->kernelReplications; i++) {
            compute_queues[i].enqueueTask(gemmkernels[i]);
        }
        for (int i=0; i < config.programSettings->kernelReplications; i++) {
            compute_queues[i].finish();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timespan = t2 - t1;
        executionTimes.push_back(timespan.count());
    }

    /* --- Read back results from Device --- */
#ifdef USE_SVM
            err = clEnqueueSVMUnmap(compute_queues[0](),
                                reinterpret_cast<void *>(a), 0,
                                NULL, NULL);
            ASSERT_CL(err)
            err = clEnqueueSVMUnmap(compute_queues[0](),
                                reinterpret_cast<void *>(b), 0,
                                NULL, NULL);
            ASSERT_CL(err)
            err = clEnqueueSVMUnmap(compute_queues[0](),
                                reinterpret_cast<void *>(c), 0,
                                NULL, NULL);
            ASSERT_CL(err)
            err = clEnqueueSVMUnmap(compute_queues[0](),
                                reinterpret_cast<void *>(c_out), 0,
                                NULL, NULL);
            ASSERT_CL(err)
#else
        // The last buffer might only contain a little bit less data 
    for (int i=0; i < config.programSettings->kernelReplications; i++) {
        long max_bytes_to_read = (static_cast<long>(sizeof(HOST_DATA_TYPE)*config.programSettings->matrixSize*config.programSettings->matrixSize))
                                            - i * sizeof(HOST_DATA_TYPE) *  out_buffer_size;
        long bytes_to_read = std::min(max_bytes_to_read, static_cast<long>(sizeof(HOST_DATA_TYPE) * out_buffer_size));
        if (bytes_to_read > 0) {
            err = compute_queues[0].enqueueReadBuffer(out_buffers[i], CL_TRUE, 0,
                                    bytes_to_read, 
                                            &c_out[i * out_buffer_size]);
            ASSERT_CL(err)
        }
    }
#endif


    std::unique_ptr<gemm::GEMMExecutionTimings> results(
                    new gemm::GEMMExecutionTimings{executionTimes});
    return results;
}

}  // namespace bm_execution
