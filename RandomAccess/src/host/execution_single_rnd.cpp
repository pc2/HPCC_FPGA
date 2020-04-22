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
#ifndef UPDATE_SPLIT
#define UPDATE_SPLIT 8
#endif

/* Related header files */
#include "src/host/execution.h"

/* C++ standard library headers */
#include <chrono>
#include <fstream>
#include <memory>
#include <vector>

/* External library headers */
#include "CL/cl.hpp"
#if QUARTUS_MAJOR_VERSION > 18
#include "CL/cl_ext_intelfpga.h"
#endif

/* Project's headers */
#include "src/host/fpga_setup.h"
#include "src/host/random_access_functionality.h"

namespace bm_execution {

    /*
    Implementation for the single_rnd kernel.
     @copydoc bm_execution::calculate()
    */
    std::shared_ptr<ExecutionResults>
    calculate(cl::Context context, cl::Device device, cl::Program program,
                   uint repetitions, uint replications, size_t dataSize,
                   bool useMemInterleaving) {
        // int used to check for OpenCL errors
        int err;
        DATA_TYPE_UNSIGNED* random;
        posix_memalign(reinterpret_cast<void **>(&random), 64,
                       sizeof(DATA_TYPE_UNSIGNED)*UPDATE_SPLIT);

        for (DATA_TYPE i=0; i < UPDATE_SPLIT; i++) {
            random[i] = starts((4 * DATA_LENGTH) / UPDATE_SPLIT * i);
        }

        std::vector<cl::CommandQueue> compute_queue;
        std::vector<cl::Buffer> Buffer_data;
        std::vector<cl::Buffer> Buffer_random;
        std::vector<cl::Kernel> accesskernel;
        std::vector<DATA_TYPE_UNSIGNED*> data_sets;

        /* --- Prepare kernels --- */

        for (int r=0; r < replications; r++) {
            DATA_TYPE_UNSIGNED* data;
            posix_memalign(reinterpret_cast<void **>(&data), 64,
                           sizeof(DATA_TYPE)*(dataSize / replications));
            data_sets.push_back(data);

            compute_queue.push_back(cl::CommandQueue(context, device));

            // Select memory bank to place data replication
            int channel = 0;
            if (!useMemInterleaving) {
                switch ((r % replications) + 1) {
                    case 1: channel = CL_CHANNEL_1_INTELFPGA; break;
                    case 2: channel = CL_CHANNEL_2_INTELFPGA; break;
                    case 3: channel = CL_CHANNEL_3_INTELFPGA; break;
                    case 4: channel = CL_CHANNEL_4_INTELFPGA; break;
                    case 5: channel = CL_CHANNEL_5_INTELFPGA; break;
                    case 6: channel = CL_CHANNEL_6_INTELFPGA; break;
                    case 7: channel = CL_CHANNEL_7_INTELFPGA; break;
                }
            }

            Buffer_data.push_back(cl::Buffer(context, channel |
                        CL_MEM_READ_WRITE,
                        sizeof(DATA_TYPE_UNSIGNED)*(dataSize / replications)));
            Buffer_random.push_back(cl::Buffer(context, channel |
                        CL_MEM_WRITE_ONLY,
                        sizeof(DATA_TYPE_UNSIGNED) * UPDATE_SPLIT));
            accesskernel.push_back(cl::Kernel(program,
                        (RANDOM_ACCESS_KERNEL + std::to_string(r)).c_str() ,
                        &err));
            ASSERT_CL(err);

            // prepare kernels
            err = accesskernel[r].setArg(0, Buffer_data[r]);
            ASSERT_CL(err);
            err = accesskernel[r].setArg(1, Buffer_random[r]);
            ASSERT_CL(err);
            err = accesskernel[r].setArg(2, DATA_TYPE_UNSIGNED(dataSize));
            ASSERT_CL(err);
            err = accesskernel[r].setArg(3,
                                DATA_TYPE_UNSIGNED(dataSize / replications));
            ASSERT_CL(err);
        }

        /* --- Execute actual benchmark kernels --- */

        double t;
        std::vector<double> executionTimes;
        for (int i = 0; i < repetitions; i++) {
            // prepare data and send them to device
            for (DATA_TYPE_UNSIGNED r =0; r < replications; r++) {
                for (DATA_TYPE_UNSIGNED j=0;
                     j < (dataSize / replications); j++) {
                    data_sets[r][j] = r*(dataSize / replications) + j;
                }
            }
            for (int r=0; r < replications; r++) {
                compute_queue[r].enqueueWriteBuffer(Buffer_data[r], CL_TRUE, 0,
                     sizeof(DATA_TYPE)*(dataSize / replications), data_sets[r]);
                 compute_queue[r].enqueueWriteBuffer(Buffer_random[r], CL_TRUE,
                     0, sizeof(DATA_TYPE_UNSIGNED) * UPDATE_SPLIT, random);
            }

            // Execute benchmark kernels
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int r=0; r < replications; r++) {
                compute_queue[r].enqueueTask(accesskernel[r]);
            }
            for (int r=0; r < replications; r++) {
                compute_queue[r].finish();
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> timespan =
                std::chrono::duration_cast<std::chrono::duration<double>>
                                                                    (t2 - t1);
            executionTimes.push_back(timespan.count());
        }

        /* --- Read back results from Device --- */

        for (int r=0; r < replications; r++) {
            compute_queue[r].enqueueReadBuffer(Buffer_data[r], CL_TRUE, 0,
                     sizeof(DATA_TYPE)*(dataSize / replications), data_sets[r]);
        }
        DATA_TYPE_UNSIGNED* data;
        posix_memalign(reinterpret_cast<void **>(&data), 64,
                                        (sizeof(DATA_TYPE)*dataSize));
        for (size_t r =0; r < replications; r++) {
            for (size_t j=0; j < (dataSize / replications); j++) {
                data[r*(dataSize / replications) + j] = data_sets[r][j];
            }
            free(reinterpret_cast<void *>(data_sets[r]));
        }

        /* --- Check Results --- */

        DATA_TYPE_UNSIGNED temp = 1;
        for (DATA_TYPE_UNSIGNED i=0; i < 4L*dataSize; i++) {
            DATA_TYPE v = 0;
            if (((DATA_TYPE)temp) < 0) {
                v = POLY;
            }
            temp = (temp << 1) ^ v;
            data[temp & (dataSize - 1)] ^= temp;
        }

        double errors = 0;
        for (DATA_TYPE_UNSIGNED i=0; i< dataSize; i++) {
            if (data[i] != i) {
                errors++;
            }
        }
        free(reinterpret_cast<void *>(data));
        free(reinterpret_cast<void *>(random));

        std::shared_ptr<ExecutionResults> results(
                        new ExecutionResults{executionTimes,
                                             errors / dataSize});
        return results;
    }

}  // namespace bm_execution
