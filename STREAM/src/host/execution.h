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
#include <complex>
#include <memory>
#include <vector>
#include <map>

/* External library headers */
#include "CL/cl.hpp"
#include "parameters.h"
#include "hpcc_benchmark.hpp"
#include "stream_functionality.hpp"

// Map keys for execution timings
#define PCIE_WRITE_KEY "PCI write"
#define PCIE_READ_KEY "PCI read"
#define COPY_KEY "Copy"
#define SCALE_KEY "Scale"
#define ADD_KEY "Add"
#define TRIAD_KEY "Triad"

namespace bm_execution {

    static std::map<std::string,double> multiplicatorMap = {
            {PCIE_WRITE_KEY, 3.0},
            {PCIE_READ_KEY, 3.0},
            {COPY_KEY, 2.0},
            {SCALE_KEY, 2.0},
            {ADD_KEY, 3.0},
            {TRIAD_KEY, 3.0}
    };

/**
The actual execution of the benchmark.
This method can be implemented in multiple *.cpp files. This header enables
simple exchange of the different calculation methods.

@param config struct that contains all necessary information to execute the kernel on the FPGA


@return The resulting matrix
*/
    std::shared_ptr<StreamExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<StreamProgramSettings> config,
              HOST_DATA_TYPE* A,
              HOST_DATA_TYPE* B,
              HOST_DATA_TYPE* C);

}  // namespace bm_execution

#endif  // SRC_HOST_EXECUTION_H_
