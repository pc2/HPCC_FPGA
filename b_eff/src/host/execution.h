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
#include <map>
#include <memory>
#include <vector>

/* External library headers */
#include "parameters.h"
#include "network_benchmark.hpp"


namespace bm_execution {

/**
The actual execution of the benchmark.
This method can be implemented in multiple *.cpp files. This header enables
simple exchange of the different calculation methods.

@param config struct that contains all necessary information to execute the kernel on the FPGA


@return The resulting matrix
*/
    std::shared_ptr<network::ExecutionTimings>
    calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings> const& config, cl_uint messageSize, cl_uint looplength, cl::vector<HOST_DATA_TYPE> &validationData);

}  // namespace bm_execution

#endif  // SRC_HOST_EXECUTION_H_
