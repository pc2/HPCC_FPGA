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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_PL_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_PL_HPP

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "mpi.h"
#include "accl.hpp"
#include "cclo_bfm.h"
#include "accl_hls.h"

/* Project's headers */

extern void send_recv(const float *read_buffer,float *write_buffer,  ap_uint<32> size, ap_uint<32> num_iterations, 
                ap_uint<32> neighbor_rank, ap_uint<32> communicator_addr, ap_uint<32> datapath_cfg,
                STREAM<command_word> &cmd, STREAM<command_word > &sts);

namespace network::execution_types::accl_pl {


    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
	template<class TDevice, class TContext, class TProgram>
    std::shared_ptr<network::ExecutionTimings>
    calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings, TDevice, TContext, TProgram> const& config, cl_uint messageSize, cl_uint looplength,
                cl::vector<HOST_DATA_TYPE> &validationData) {

        int err;
        std::vector<cl::vector<HOST_DATA_TYPE>> dummyBufferContents;
        std::vector<cl::vector<HOST_DATA_TYPE>> recvBufferContents;
	std::vector<std::unique_ptr<ACCL::Buffer<HOST_DATA_TYPE>>> acclSendBuffers;
	std::vector<std::unique_ptr<ACCL::Buffer<HOST_DATA_TYPE>>> acclRecvBuffers;
        cl_uint size_in_bytes = std::max(static_cast<int>(validationData.size()), (1 << messageSize));

        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

        hlslib::Stream<stream_word> cclo2krnl("cclo2krnl"), krnl2cclo("krnl2cclo");
        hlslib::Stream<command_word> cmd, sts;

        std::vector<unsigned int> dest = {0};
        std::unique_ptr<CCLO_BFM> cclo;
        if (config.programSettings->useAcclEmulation) {
            cclo = std::make_unique<CCLO_BFM>(6000, current_rank, current_size, dest, cmd, sts, cclo2krnl, krnl2cclo);
            cclo->run();
        }
        MPI_Barrier(MPI_COMM_WORLD);

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
                acclSendBuffers.push_back(config.accl->create_buffer(dummyBufferContents.back().data(), size_in_values * 4, ACCL::dataType::float32));
                acclRecvBuffers.push_back(config.accl->create_buffer(recvBufferContents.back().data(), size_in_values * 4, ACCL::dataType::float32));
                acclSendBuffers.back()->sync_to_device();
                acclRecvBuffers.back()->sync_to_device();
            }

            xrt::kernel sendrecvKernel;
            if (!config.programSettings->useAcclEmulation) {
                sendrecvKernel(*config.device, *config.program, "send_recv");
            }

            double calculationTime = 0.0;
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                auto startCalculation = std::chrono::high_resolution_clock::now();
                if (!config.programSettings->useAcclEmulation) {
                auto run = sendrecvKernel(acclSendBuffers[i]->bo(), acclRecvBuffers[i]->bo(), size_in_values, looplength, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                            config.accl->get_communicator_addr(), config.accl->get_arithmetic_config_addr({ACCL::dataType::float32, ACCL::dataType::float32}));
                run.wait();
                } else {
                    send_recv(reinterpret_cast<float*>(acclSendBuffers[i]->buffer()), reinterpret_cast<float*>(acclRecvBuffers[i]->buffer()), size_in_values, looplength, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                            config.accl->get_communicator_addr(), config.accl->get_arithmetic_config_addr({ACCL::dataType::float32, ACCL::dataType::float32}),
                                            cmd, sts);
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

        if (config.programSettings->useAcclEmulation) {
            cclo->stop();
        }
        // Read validation data from FPGA will be placed sequentially in buffer for all replications
        // The data order should not matter, because every byte should have the same value!
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {
            if (!config.programSettings->useAcclEmulation) {
                acclRecvBuffers.back()->sync_from_device();
            }
		    std::copy(recvBufferContents[r].begin(), recvBufferContents[r].begin() + validationData.size() / config.programSettings->kernelReplications, validationData.begin() + validationData.size() / config.programSettings->kernelReplications * r);
        }
        std::shared_ptr<network::ExecutionTimings> result(new network::ExecutionTimings{
                looplength,
                messageSize,
                calculationTimings
        });
        return result;
    }

}  // namespace bm_execution

#endif
