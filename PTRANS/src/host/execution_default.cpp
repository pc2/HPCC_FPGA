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
    std::shared_ptr<ExecutionTimings>
    calculate(std::shared_ptr<ExecutionConfiguration> config, HOST_DATA_TYPE *const A,
              HOST_DATA_TYPE *const B, HOST_DATA_TYPE *A_out) {

        cl::Buffer bufferA(config->context, CL_MEM_READ_ONLY,
                           sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);
        cl::Buffer bufferB(config->context, CL_MEM_READ_ONLY,
                           sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);
        cl::Buffer bufferA_out(config->context, CL_MEM_WRITE_ONLY,
                               sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);

        cl::Kernel transposeKernel(config->program, config->kernelName.c_str());

        transposeKernel.setArg(0, bufferA);
        transposeKernel.setArg(1, bufferB);
        transposeKernel.setArg(2, bufferA_out);
        transposeKernel.setArg(3, config->matrixSize);

        cl::CommandQueue queue(config->context);

        std::vector<double> transferTimings;
        std::vector<double> calculationTimings;

        for (int repetition = 0; repetition < config->repetitons; repetition++) {

            auto startTransfer = std::chrono::high_resolution_clock::now();
            queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0,
                                     sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize, A);
            queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0,
                                     sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize, B);
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
            queue.enqueueReadBuffer(bufferA_out, CL_TRUE, 0,
                                    sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize, A_out);
            endTransfer = std::chrono::high_resolution_clock::now();
            transferTime +=
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);
            transferTimings.push_back(transferTime.count());
        }

        std::shared_ptr<ExecutionTimings> result(new ExecutionTimings{
                transferTimings,
                calculationTimings
        });
        return result;
    }

}  // namespace bm_execution
