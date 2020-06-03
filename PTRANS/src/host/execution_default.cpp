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
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& config, HOST_DATA_TYPE *const A,
              HOST_DATA_TYPE *const B, HOST_DATA_TYPE *A_out) {

        cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY,
                           sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize);
        cl::Buffer bufferB(*config.context, CL_MEM_READ_ONLY,
                           sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize);
        cl::Buffer bufferA_out(*config.context, CL_MEM_WRITE_ONLY,
                               sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize);

        cl::Kernel transposeKernel(*config.program, KERNEL_NAME);

#ifdef USE_SVM
        clSetKernelArgSVMPointer(transposeKernel(), 0,
                                        reinterpret_cast<void*>(A));
        clSetKernelArgSVMPointer(transposeKernel(), 1,
                                        reinterpret_cast<void*>(B));
        clSetKernelArgSVMPointer(transposeKernel(), 2,
                                        reinterpret_cast<void*>(A_out));
#else
        transposeKernel.setArg(0, bufferA);
        transposeKernel.setArg(1, bufferB);
        transposeKernel.setArg(2, bufferA_out);
#endif
        transposeKernel.setArg(3, config.programSettings->matrixSize / config.programSettings->blockSize);

        cl::CommandQueue queue(*config.context);

        std::vector<double> transferTimings;
        std::vector<double> calculationTimings;

        for (int repetition = 0; repetition < config.programSettings->numRepetitions; repetition++) {

            auto startTransfer = std::chrono::high_resolution_clock::now();
#ifdef USE_SVM
        clEnqueueSVMMap(queue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(A),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        clEnqueueSVMMap(queue(), CL_TRUE,
                        CL_MAP_READ,
                        reinterpret_cast<void *>(B),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
        clEnqueueSVMMap(queue(), CL_TRUE,
                        CL_MAP_WRITE,
                        reinterpret_cast<void *>(A_out),
                        sizeof(HOST_DATA_TYPE) *
                        (config.programSettings->matrixSize * config.programSettings->matrixSize), 0,
                        NULL, NULL);
#else
            queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0,
                                     sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize, A);
            queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0,
                                     sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize, B);
#endif
            queue.finish();
            auto endTransfer = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> transferTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);

            auto startCalculation = std::chrono::high_resolution_clock::now();
            queue.enqueueTask(transposeKernel);
            queue.finish();
            auto endCalculation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());

            startTransfer = std::chrono::high_resolution_clock::now();
#ifdef USE_SVM
            clEnqueueSVMUnmap(queue(),
                                reinterpret_cast<void *>(A), 0,
                                NULL, NULL);
            clEnqueueSVMUnmap(queue(),
                                reinterpret_cast<void *>(B), 0,
                                NULL, NULL);
            clEnqueueSVMUnmap(queue(),
                                reinterpret_cast<void *>(A_out), 0,
                                NULL, NULL);
#else
            queue.enqueueReadBuffer(bufferA_out, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE) * config.programSettings->matrixSize * config.programSettings->matrixSize, A_out);
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
