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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_PL_STREAM_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_ACCL_PL_STREAM_HPP

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

/* External library headers */
#include "mpi.h"
#include "accl.hpp"
#include "cclo_bfm.h"
#include "accl_hls.h"

/* Project's headers */
typedef ap_axiu<1, 0, 0, 0> notify_word;

extern void send_stream(ap_uint<512>* read_buffer, ap_uint<32> size, ap_uint<32> num_iterations, 
                        STREAM<stream_word > &data_out);

extern void recv_stream(ap_uint<512>* write_buffer, ap_uint<32> size, ap_uint<32> num_iterations, 
                STREAM<stream_word> &data_in, STREAM<notify_word> &notify);

extern void schedule_stream(ap_uint<32> size, ap_uint<32> num_iterations, 
                ap_uint<32> neighbor_rank, ap_uint<32> communicator_addr, ap_uint<32> datapath_cfg,
                STREAM<command_word> &cmd, STREAM<command_word > &sts, STREAM<notify_word> &notify);

namespace network::execution_types::accl_pl_stream {


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
        cl_uint size_in_bytes = (1 << messageSize);

        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

        hlslib::Stream<stream_word> cclo2krnl("cclo2krnl"), krnl2cclo("krnl2cclo");
        hlslib::Stream<command_word> cmd("cmd"), sts("sts");
        hlslib::Stream<notify_word> notify("notify");

        std::vector<unsigned int> dest = {0};
        std::unique_ptr<CCLO_BFM> cclo;
        if (config.programSettings->useAcclEmulation) {
            cclo = std::make_unique<CCLO_BFM>(6000, current_rank, current_size, dest, cmd, sts, cclo2krnl, krnl2cclo);
            cclo->run();
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<double> calculationTimings;
        for (uint r =0; r < config.programSettings->numRepetitions; r++) {
            acclSendBuffers.clear();
            acclRecvBuffers.clear();
            dummyBufferContents.clear();
            recvBufferContents.clear();
            int size_in_values = (size_in_bytes + 3) / 4;

            xrt::kernel sendKernel;
            xrt::kernel recvKernel;
            xrt::kernel scheduleKernel;
            if (!config.programSettings->useAcclEmulation) {
                sendKernel = xrt::kernel(*config.device, *config.program, "send_stream");
                recvKernel = xrt::kernel(*config.device, *config.program, "recv_stream");
                scheduleKernel = xrt::kernel(*config.device, *config.program, "schedule_stream");
            }
            // Create all kernels and buffers. The kernel pairs are generated twice to utilize all channels
            for (int r = 0; r < config.programSettings->kernelReplications; r++) {
                dummyBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(messageSize & (255)));
                recvBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(0));
                if (config.programSettings->useAcclEmulation) {
                    acclSendBuffers.push_back(config.context->accl->create_buffer(dummyBufferContents.back().data(), size_in_bytes, ACCL::dataType::int32, 0));
                    acclRecvBuffers.push_back(config.context->accl->create_buffer(recvBufferContents.back().data(), size_in_bytes, ACCL::dataType::int32, 1));
                }
                else {
                    acclSendBuffers.push_back(config.context->accl->create_buffer(dummyBufferContents.back().data(), size_in_bytes, ACCL::dataType::int32, sendKernel.group_id(0)));
                    acclRecvBuffers.push_back(config.context->accl->create_buffer(recvBufferContents.back().data(), size_in_bytes, ACCL::dataType::int32, recvKernel.group_id(0)));               
                }
                acclSendBuffers.back()->sync_to_device();
                acclRecvBuffers.back()->sync_to_device();
            }

            double calculationTime = 0.0;
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                auto startCalculation = std::chrono::high_resolution_clock::now();
                if (!config.programSettings->useAcclEmulation) {
                    auto run_recv = recvKernel(*acclRecvBuffers[i]->bo(), size_in_values, looplength);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    auto run_send = sendKernel(*acclSendBuffers[i]->bo(), size_in_values, looplength);
                    MPI_Barrier(MPI_COMM_WORLD);
                    startCalculation = std::chrono::high_resolution_clock::now();
                    auto run_schedule = scheduleKernel(size_in_values, looplength, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                            config.context->accl->get_communicator_addr(), config.context->accl->get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32}));
                    run_send.wait();
                    run_recv.wait();
                    run_schedule.wait();
                } else {
                    std::thread run_send(send_stream, reinterpret_cast<ap_uint<512>*>(acclSendBuffers[i]->buffer()), size_in_values, looplength,
                                            std::ref(krnl2cclo));
                    std::thread run_recv(recv_stream, reinterpret_cast<ap_uint<512>*>(acclRecvBuffers[i]->buffer()), size_in_values, looplength,
                                            std::ref(cclo2krnl), std::ref(notify));
                    std::thread run_schedule(schedule_stream,size_in_values, looplength, (current_rank - 1 + 2 * ((current_rank + i) % 2) + current_size) % current_size,
                                            config.context->accl->get_communicator_addr(), config.context->accl->get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32}),
                                            std::ref(cmd), std::ref(sts), std::ref(notify));
                    run_send.join();
                    run_recv.join();
                    run_schedule.join();
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
                acclRecvBuffers[r]->sync_from_device();
            }
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
