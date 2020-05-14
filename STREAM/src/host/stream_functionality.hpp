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

#ifndef SRC_HOST_STREAM_BENCHMARK_H_
#define SRC_HOST_STREAM_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"


class StreamProgramSettings : public hpcc_base::BaseSettings {

public:
    uint streamArraySize;
    uint kernelReplications;
    bool useSingleKernel;

    StreamProgramSettings(cxxopts::ParseResult &results);

};

class StreamData {

public:
    HOST_DATA_TYPE *A, *B, *C;
    StreamData(HOST_DATA_TYPE *A,HOST_DATA_TYPE *B,HOST_DATA_TYPE *C) : A(A), B(B), C(C) {}
    StreamData(StreamData *d) : A(d->A), B(d->B), C(d->C) {}

};

class StreamExecutionTimings {
public:
        std::map<std::string,std::vector<double>> timings;
        uint arraySize;
};

class StreamBenchmark : public hpcc_base::HpccFpgaBenchmark<StreamProgramSettings, StreamData, StreamExecutionTimings> {

protected:

    std::shared_ptr<StreamData>
    generateInputData(const hpcc_base::ExecutionSettings<StreamProgramSettings> &settings) override;

    std::shared_ptr<StreamExecutionTimings>
    executeKernel(const hpcc_base::ExecutionSettings<StreamProgramSettings> &settings, StreamData &data) override;

    bool
    validateOutputAndPrintError(const hpcc_base::ExecutionSettings<StreamProgramSettings> &settings ,StreamData &data, const StreamExecutionTimings &output) override;

    void
    printResults(const hpcc_base::ExecutionSettings<StreamProgramSettings> &settings, const StreamExecutionTimings &output) override;

    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    StreamBenchmark(int argc, char* argv[]);

};


#endif // SRC_HOST_STREAM_BENCHMARK_H_
