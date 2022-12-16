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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_PCIE_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_PCIE_HPP

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "mpi.h"

/* Project's headers */

namespace network::execution_types::pcie {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    network::ExecutionTimings
    calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings> const& config, cl_uint messageSize, cl_uint looplength,
                cl::vector<HOST_DATA_TYPE> &validationData) {

        int err;
        std::vector<cl::CommandQueue> sendQueues;
        std::vector<cl::Buffer> dummyBuffers;
        std::vector<cl::vector<HOST_DATA_TYPE>> dummyBufferContents;

        cl_uint size_in_bytes = std::max(static_cast<int>(validationData.size()), (1 << messageSize));

        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

        std::vector<double> calculationTimings;
        for (uint r =0; r < config.programSettings->numRepetitions; r++) {
            sendQueues.clear();
            dummyBuffers.clear();
            dummyBufferContents.clear();
            // Create all kernels and buffers. The kernel pairs are generated twice to utilize all channels
            for (int r = 0; r < config.programSettings->kernelReplications; r++) {

                dummyBuffers.push_back(cl::Buffer(*config.context, CL_MEM_READ_WRITE, sizeof(HOST_DATA_TYPE) * size_in_bytes,0,&err));
                ASSERT_CL(err)

                dummyBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(messageSize & (255)));

                cl::CommandQueue sendQueue(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)

                sendQueue.enqueueWriteBuffer(dummyBuffers.back(), CL_TRUE, 0, sizeof(HOST_DATA_TYPE) * size_in_bytes, dummyBufferContents.back().data());

                sendQueues.push_back(sendQueue);

            }
            double calculationTime = 0.0;
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                auto startCalculation = std::chrono::high_resolution_clock::now();
                for (int l = 0; l < looplength; l++) {

                        sendQueues[i].enqueueReadBuffer(dummyBuffers[i], CL_TRUE, 0, sizeof(HOST_DATA_TYPE) * size_in_bytes, dummyBufferContents[i].data());

                        MPI_Sendrecv(dummyBufferContents[i].data(), size_in_bytes, MPI_CHAR, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size, 0, 
                                        dummyBufferContents[i].data(), size_in_bytes, MPI_CHAR, (current_rank - 1 + 2 * ((current_rank + i) % 2)  + current_size) % current_size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        sendQueues[i].enqueueWriteBuffer(dummyBuffers[i], CL_TRUE, 0, sizeof(HOST_DATA_TYPE) * size_in_bytes, dummyBufferContents[i].data());

                }
                auto endCalculation = std::chrono::high_resolution_clock::now();
                calculationTime += std::chrono::duration_cast<std::chrono::duration<double>>(endCalculation - startCalculation).count();
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Enqueued " << r << "," << i << std::endl;
                #endif
            }
            calculationTimings.push_back(calculationTime);
#ifndef NDEBUG
        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
        std::cout << "Rank " << current_rank << ": Done " << r << std::endl;
#endif
        }
        // Read validation data from FPGA will be placed sequentially in buffer for all replications
        // The data order should not matter, because every byte should have the same value!
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {
            err = sendQueues[r].enqueueReadBuffer(dummyBuffers[r], CL_TRUE, 0, sizeof(HOST_DATA_TYPE) * validationData.size() / config.programSettings->kernelReplications, &validationData.data()[r * validationData.size() / config.programSettings->kernelReplications]);
            ASSERT_CL(err);
        }
        return network::ExecutionTimings{
                looplength,
                messageSize,
                calculationTimings
        };
    }

}  // namespace bm_execution

#endif
