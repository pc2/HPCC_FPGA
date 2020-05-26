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
    std::unique_ptr<fft::FFTExecutionTimings>
    calculate(hpcc_base::ExecutionSettings<fft::FFTProgramSettings> const&  config,
            std::complex<HOST_DATA_TYPE>* data,
            unsigned iterations,
            bool inverse) {

        cl::Buffer inBuffer = cl::Buffer(*config.context, CL_MEM_WRITE_ONLY, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE));
        cl::Buffer outBuffer = cl::Buffer(*config.context, CL_MEM_READ_ONLY, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE));

        cl::Kernel fetchKernel(*config.program, FETCH_KERNEL_NAME);
        cl::Kernel fftKernel(*config.program, FFT_KERNEL_NAME);

#ifdef USE_SVM
        err = clSetKernelArgSVMPointer(fetchkernel(), 0,
                                        reinterpret_cast<void*>(data));
        err = clSetKernelArgSVMPointer(fftkernel(), 0,
                                        reinterpret_cast<void*>(data));
#else
        fetchKernel.setArg(0, inBuffer);
        fftKernel.setArg(0, outBuffer);
#endif
        fftKernel.setArg(1, iterations);
        fftKernel.setArg(2, static_cast<cl_int>(inverse));

        cl::CommandQueue fetchQueue(*config.context);
        cl::CommandQueue fftQueue(*config.context);

#ifdef USE_SVM
        clEnqueueSVMMap(compute_queue(), CL_TRUE,
                        CL_MAP_READ | CL_MAP_WRITE,
                        reinterpret_cast<void *>(data),
                        (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE), 0,
                        NULL, NULL);
#else
        fetchQueue.enqueueWriteBuffer(inBuffer,CL_TRUE,0, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE), data);
#endif

        std::vector<double> calculationTimings;
        for (uint r =0; r < config.programSettings->numRepetitions; r++) {
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
#ifdef USE_SVM
            clEnqueueSVMUnmap(compute_queue(),
                                reinterpret_cast<void *>(data), 0,
                                NULL, NULL);
#else
        fetchQueue.enqueueReadBuffer(outBuffer,CL_TRUE,0, (1 << LOG_FFT_SIZE) * iterations * 2 * sizeof(HOST_DATA_TYPE), data);
#endif

        std::unique_ptr<fft::FFTExecutionTimings> result(new fft::FFTExecutionTimings{
                calculationTimings
        });
        return result;
    }

}  // namespace bm_execution
