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
#include "transpose_benchmark.hpp"


namespace bm_execution {


/**
 * @brief Transpose and add the matrices using the OpenCL kernel
 * 
 * @param config The progrma configuration
 * @param A The input matrix A
 * @param B The input matrix B
 * @param A_out The matrix storing the result of the calculation
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured executuon times 
 */
    std::unique_ptr<transpose::TransposeExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& config, HOST_DATA_TYPE *const A,
              HOST_DATA_TYPE *const B, HOST_DATA_TYPE *A_out);

}  // namespace bm_execution

#endif  // SRC_HOST_EXECUTION_H_
