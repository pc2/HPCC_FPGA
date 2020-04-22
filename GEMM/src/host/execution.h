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
#ifndef SRC_HOST_EXECUTION_H_
#define SRC_HOST_EXECUTION_H_

/* C++ standard library headers */
#include <memory>
#include <vector>

/* External library headers */
#include "CL/cl.hpp"

#include "parameters.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif


namespace bm_execution {

    struct ExecutionConfiguration {
        cl::Context context;
        cl::Device device;
        cl::Program program;
        std::string kernelName;
        uint repetitions;
        cl_uint matrixSize;
        bool useMemInterleaving;
    };

    struct ExecutionTimings {
        std::vector<double> transferTimings;
        std::vector<double> calculationTimings;
    };


/**
The actual execution of the benchmark.
This method can be implemented in multiple *.cpp files. This header enables
simple exchange of the different calculation methods.

@param context OpenCL context used to create needed Buffers and queues
@param device The OpenCL device that is used to execute the benchmarks
@param program The OpenCL program containing the kernels
@param repetitions Number of times the kernels are executed
@param replications Number of times a kernel is replicated - may be used in
                    different ways depending on the implementation of this
                    method
@param dataSize The size of the data array that may be used for benchmark
                execution in number of items
@param blockSize Size of a block that is calculated by the kernel

@return The time measurements and the error rate counted from the executions
*/
std::shared_ptr<ExecutionTimings>
calculate(std::shared_ptr<ExecutionConfiguration> config, HOST_DATA_TYPE* a, HOST_DATA_TYPE* b, HOST_DATA_TYPE* c,
        HOST_DATA_TYPE* c_out, HOST_DATA_TYPE alpha, HOST_DATA_TYPE beta);
}  // namespace bm_execution

#endif  // SRC_HOST_EXECUTION_H_
