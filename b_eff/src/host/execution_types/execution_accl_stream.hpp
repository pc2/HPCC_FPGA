/*
Copyright (c) 2022 Marius Meyer

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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_STREAM_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_STREAM_HPP

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "mpi.h"
#include "accl.hpp"

/* Project's headers */

namespace network::execution_types::accl_stream {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
	template<class TDevice, class TContext, class TProgram>
    network::ExecutionTimings
    calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings, TDevice, TContext, TProgram> const& config, cl_uint messageSize, cl_uint looplength,
                cl::vector<HOST_DATA_TYPE> &validationData) {

        int err;
        std::vector<cl::vector<HOST_DATA_TYPE>> dummyBufferContents;
        std::vector<cl::vector<HOST_DATA_TYPE>> recvBufferContents;
	    std::vector<std::unique_ptr<ACCL::Buffer<HOST_DATA_TYPE>>> acclSendBuffers;
	    std::vector<std::unique_ptr<ACCL::Buffer<HOST_DATA_TYPE>>> acclRecvBuffers;
        size_t size_in_bytes = std::max((1 << messageSize), 4);

        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

        std::vector<double> calculationTimings;
        for (uint r =0; r < config.programSettings->numRepetitions; r++) {
            dummyBufferContents.clear();
	    recvBufferContents.clear();
	    acclSendBuffers.clear();
	    acclRecvBuffers.clear();
	    int size_in_values = (size_in_bytes + 3) / 4;
            // Create all kernels and buffers. The kernel pairs are generated twice to utilize all channels
            for (int r = 0; r < config.programSettings->kernelReplications; r++) {
                dummyBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(messageSize & (255)));
                recvBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(0));
		        acclSendBuffers.push_back(config.context->accl->create_buffer(dummyBufferContents.back().data(), size_in_bytes, ACCL::dataType::float32, 0));
		        acclRecvBuffers.push_back(config.context->accl->create_buffer(recvBufferContents.back().data(), size_in_bytes, ACCL::dataType::float32, 1));
		        acclSendBuffers.back()->sync_to_device();
		        acclRecvBuffers.back()->sync_to_device();
            }

            double calculationTime = 0.0;
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                auto run_recv = recvKernel(*acclRecvBuffers[i]->bo(), size_in_values, looplength);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                MPI_Barrier(MPI_COMM_WORLD);
                auto run_send = sendKernel(*acclSendBuffers[i]->bo(), size_in_values, looplength);
                auto startCalculation = std::chrono::high_resolution_clock::now();
                auto run_schedule = scheduleKernel(size_in_values, looplength, 0, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                        config.context->accl->get_communicator_addr(), config.context->accl->get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32}));
                for (int l = 0; l < looplength; l++) {
#ifndef NDEBUG
                    std::cout << "Stream " << size_in_bytes << " bytes to " 
                                << ((current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size) << std::endl;
#endif
            config.context->accl->stream_put(*acclSendBuffers[i], size_in_values, 
                                        (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                        0, ACCL::GLOBAL_COMM, true);
#ifndef NDEBUG
                    std::cout << "Done" << std::endl;
#endif
                }
                run_send.wait();
                run_recv.wait();
                run_schedule.wait();
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
            acclRecvBuffers[r]->sync_from_device();
		    std::copy(recvBufferContents[r].begin(), recvBufferContents[r].end(), &validationData.data()[size_in_bytes * r]);
        }
        return network::ExecutionTimings{
               looplength,
                messageSize,
                calculationTimings
        };
    }

}  // namespace bm_execution

#endif
