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
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "CL/cl.hpp"

/* Project's headers */

namespace bm_execution {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    std::unique_ptr<transpose::TransposeExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& config, transpose::TransposeData& data) {
        int err;

        size_t buffer_size = data.blockSize * (data.blockSize * data.numBlocks) * sizeof(HOST_DATA_TYPE);

        // TODO add support for multiple kernel replications here
        if (config.programSettings->kernelReplications > 1) {
                std::cerr << "WARNING: Will only use a single kernel for execution since multi kernel support is missing" << std::endl;
        }

        cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY,
                           buffer_size);
        cl::Buffer bufferB(*config.context, CL_MEM_READ_ONLY,
                           buffer_size);
        cl::Buffer bufferA_out(*config.context, CL_MEM_WRITE_ONLY,
                               buffer_size);

        cl::Kernel transposeReadKernel(*config.program, (READ_KERNEL_NAME + std::to_string(0)).c_str(), &err);
        ASSERT_CL(err)
        cl::Kernel transposeWriteKernel(*config.program, (WRITE_KERNEL_NAME + std::to_string(0)).c_str(), &err);
        ASSERT_CL(err)

#ifdef USE_SVM
        err = clSetKernelArgSVMPointer(transposeReadKernel(), 0,
                                        reinterpret_cast<void*>(data.A));
        ASSERT_CL(err)
        err = clSetKernelArgSVMPointer(transposeWriteKernel(), 0,
                                        reinterpret_cast<void*>(data.B));
        ASSERT_CL(err)
        err = clSetKernelArgSVMPointer(transposeWriteKernel(), 1,
                                        reinterpret_cast<void*>(data.result));
        ASSERT_CL(err)
#else
        err = transposeReadKernel.setArg(0, bufferA);
        ASSERT_CL(err)   
        err = transposeWriteKernel.setArg(0, bufferB);
        ASSERT_CL(err)
        err = transposeWriteKernel.setArg(1, bufferA_out);
        ASSERT_CL(err)
 
#endif
        err = transposeWriteKernel.setArg(2, 0);
        ASSERT_CL(err) 
        err = transposeReadKernel.setArg(1, 0);
        ASSERT_CL(err) 
        err = transposeWriteKernel.setArg(3, data.numBlocks);
        ASSERT_CL(err) 
        err = transposeReadKernel.setArg(2, data.numBlocks);
        ASSERT_CL(err)     

        cl::CommandQueue readQueue(*config.context, *config.device, 0, &err);
        ASSERT_CL(err)
        cl::CommandQueue writeQueue(*config.context, *config.device, 0, &err);
        ASSERT_CL(err)

        std::vector<double> transferTimings;
        std::vector<double> calculationTimings;

        for (int repetition = 0; repetition < config.programSettings->numRepetitions; repetition++) {

            auto startTransfer = std::chrono::high_resolution_clock::now();
#ifdef USE_SVM
        clEnqueueSVMMap(readQueue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(data.A),
                        buffer_size, 0,
                        NULL, NULL);
        clEnqueueSVMMap(writeQueue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(data.B),
                        buffer_size, 0,
                        NULL, NULL);
        clEnqueueSVMMap(writeQueue(), CL_TRUE,
                        CL_MAP_WRITE,
                        reinterpret_cast<void *>(data.result),
                        buffer_size, 0,
                        NULL, NULL);
#else
            readQueue.enqueueWriteBuffer(bufferA, CL_FALSE, 0,
                                     buffer_size, data.A);
            writeQueue.enqueueWriteBuffer(bufferB, CL_FALSE, 0,
                                     buffer_size, data.B);
#endif
            writeQueue.finish();
            readQueue.finish();
            auto endTransfer = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> transferTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);

            auto startCalculation = std::chrono::high_resolution_clock::now();
            writeQueue.enqueueTask(transposeWriteKernel);
            readQueue.enqueueTask(transposeReadKernel);
            writeQueue.finish();
            readQueue.finish();
            auto endCalculation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());

            startTransfer = std::chrono::high_resolution_clock::now();
#ifdef USE_SVM
            clEnqueueSVMUnmap(readQueue(),
                                reinterpret_cast<void *>(data.A), 0,
                                NULL, NULL);
            clEnqueueSVMUnmap(writeQueue(),
                                reinterpret_cast<void *>(data.B), 0,
                                NULL, NULL);
            clEnqueueSVMUnmap(writeQueue(),
                                reinterpret_cast<void *>(data.result), 0,
                                NULL, NULL);
#else
            writeQueue.enqueueReadBuffer(bufferA_out, CL_TRUE, 0,
                                    buffer_size, data.result);
#endif
            endTransfer = std::chrono::high_resolution_clock::now();
            transferTime +=
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);
            transferTimings.push_back(transferTime.count());
        }

        std::unique_ptr<transpose::TransposeExecutionTimings> result(new transpose::TransposeExecutionTimings{
                transferTimings,
                calculationTimings
        });
        return result;
    }

}  // namespace bm_execution
