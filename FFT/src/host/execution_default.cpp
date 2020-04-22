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
#include "CL/cl_ext_intelfpga.h"

/* Project's headers */

namespace bm_execution {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    std::shared_ptr<ExecutionTimings>
    calculate(std::shared_ptr<ExecutionConfiguration> config,
            std::complex<HOST_DATA_TYPE>* data,
            unsigned iterations,
            bool inverse) {

        cl::Buffer inBuffer = cl::Buffer(config->context, CL_MEM_WRITE_ONLY, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE));
        cl::Buffer outBuffer = cl::Buffer(config->context, CL_MEM_READ_ONLY, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE));

        cl::Kernel fetchKernel(config->program, FETCH_KERNEL_NAME);

        fetchKernel.setArg(0, inBuffer);

        cl::Kernel fftKernel(config->program, FFT_KERNEL_NAME);

        fftKernel.setArg(0, outBuffer);
        fftKernel.setArg(1, iterations);
        fftKernel.setArg(2, static_cast<cl_int>(inverse));

        cl::CommandQueue fetchQueue(config->context);
        cl::CommandQueue fftQueue(config->context);

        fetchQueue.enqueueWriteBuffer(inBuffer,CL_TRUE,0, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE), data);

        std::vector<double> calculationTimings;
        for (uint r =0; r < config->repetitions; r++) {
            auto startCalculation = std::chrono::high_resolution_clock::now();
            fetchQueue.enqueueNDRangeKernel(fetchKernel, cl::NullRange, cl::NDRange((1 << LOG_FFT_SIZE)/ FFT_UNROLL * iterations),
                    cl::NDRange((1 << LOG_FFT_SIZE)/ FFT_UNROLL));
            fftQueue.enqueueTask(fftKernel);
            fetchQueue.finish();
            fftQueue.finish();
            auto endCalculation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());
        }

        fetchQueue.enqueueReadBuffer(outBuffer,CL_TRUE,0, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE), data);

        std::shared_ptr<ExecutionTimings> result(new ExecutionTimings{
                iterations,
                inverse,
                calculationTimings
        });
        return result;
    }

}  // namespace bm_execution
