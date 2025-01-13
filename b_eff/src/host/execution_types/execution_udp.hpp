/*
Copyright (c) 2024 Marius Meyer

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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_UDP_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_UDP_HPP

/* C++ standard library headers */
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

/* External library headers */
#include "experimental/xrt_kernel.h"
#include "mpi.h"

// /* Project's headers */
// typedef ap_axiu<1, 0, 0, 0> notify_word;

// extern void send_stream(ap_uint<512> *read_buffer, ap_uint<32> size, ap_uint<32> num_iterations,
//                         STREAM<stream_word> &data_out, STREAM<notify_word> &notify);

// extern void recv_stream(ap_uint<512> *write_buffer, ap_uint<32> size, ap_uint<32> num_iterations,
//                         STREAM<stream_word> &data_in, STREAM<notify_word> &notify);

namespace network::execution_types::udp
{

/*
Implementation for the single kernel.
 @copydoc bm_execution::calculate()
*/
template <class TDevice, class TContext, class TProgram>
network::ExecutionTimings
calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings, TDevice, TContext, TProgram> const &config,
          cl_uint messageSize, cl_uint looplength, cl::vector<HOST_DATA_TYPE> &validationData)
{

    int err;
    std::vector<cl::vector<HOST_DATA_TYPE>> dummyBufferContents;
    std::vector<cl::vector<HOST_DATA_TYPE>> recvBufferContents;
    std::vector<xrt::bo> boSend;
    std::vector<xrt::bo> boRecv;
    std::vector<xrt::kernel> sendKernels;
    std::vector<xrt::kernel> recvKernels;
    cl_uint size_in_bytes = (1 << messageSize);

    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    int current_size;
    MPI_Comm_size(MPI_COMM_WORLD, &current_size);

    // hlslib::Stream<stream_word> udp_emu("udp_emu");
    // hlslib::Stream<command_word> cmd("cmd"), sts("sts");
    // hlslib::Stream<notify_word> notify("notify");

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> calculationTimings;
    for (uint r = 0; r < config.programSettings->numRepetitions; r++) {
        dummyBufferContents.clear();
        recvBufferContents.clear();
        sendKernels.clear();
        recvKernels.clear();
        boSend.clear();
        boRecv.clear();
        int size_in_values = (size_in_bytes + 3) / 4;

        // Create all kernels and buffers. The kernel pairs are generated twice to utilize all channels
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {
            dummyBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(messageSize & (255)));
            recvBufferContents.emplace_back(size_in_bytes, static_cast<HOST_DATA_TYPE>(0));
            sendKernels.emplace_back(*config.device, *config.program,
                                     "send_stream:{send_stream_" + std::to_string(r) + "}");
            recvKernels.emplace_back(*config.device, *config.program,
                                     "recv_stream:{recv_stream_" + std::to_string(r) + "}");
            boSend.emplace_back(*config.device, dummyBufferContents[r].data(), size_in_bytes,
                                sendKernels[r].group_id(0));
            boRecv.emplace_back(*config.device, recvBufferContents[r].data(), size_in_bytes,
                                recvKernels[r].group_id(0));

            boSend[r].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        double calculationTime = 0.0;
        std::vector<xrt::run> sendRuns;
        std::vector<xrt::run> recvRuns;
        for (int i = 0; i < config.programSettings->kernelReplications; i++) {
            recvRuns.push_back(recvKernels[i](boRecv[i], size_in_values, looplength, 1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        MPI_Barrier(MPI_COMM_WORLD);
        auto startCalculation = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.programSettings->kernelReplications; i++) {
            // std::cout << i << " to "
            //           << ((current_rank + i * current_size + 1) %
            //               (current_size * config.programSettings->kernelReplications))
            //           << std::endl;
            sendRuns.push_back(
                sendKernels[i](boSend[i], size_in_values, looplength, 0, config.programSettings->payload_size));
        }
        // std::cout << "Wait for the send kernels..." << std::endl;
        for (auto &r : sendRuns) {
            r.wait();
            //     auto state = ERT_CMD_STATE_TIMEOUT;
            //     // while (state == ERT_CMD_STATE_TIMEOUT) {
            //     state = r.wait(std::chrono::seconds(2));
            //     for (auto &u : config.context->udps) {
            //         u.print_socket_table(4);
            //         std::cout << "Stats " << std::endl;
            //         u.get_udp_out_pkts();
            //         u.get_udp_in_pkts();
            //         u.get_udp_app_in_pkts();
            //     }
            //     // }
        }
        // std::cout << "Wait for the recv kernels..." << std::endl;
        for (auto &r : recvRuns) {
            r.wait();
            //     auto state = ERT_CMD_STATE_TIMEOUT;
            //     // while (state == ERT_CMD_STATE_TIMEOUT) {
            //     state = r.wait(std::chrono::seconds(2));
            //     for (auto &u : config.context->udps) {
            //         std::cout << "Stats " << std::endl;
            //         u.get_udp_out_pkts();
            //         u.get_udp_in_pkts();
            //         u.get_udp_app_in_pkts();
            //     }
            //     // }
        }
        auto endCalculation = std::chrono::high_resolution_clock::now();
        calculationTime +=
            std::chrono::duration_cast<std::chrono::duration<double>>(endCalculation - startCalculation).count();
        calculationTimings.push_back(calculationTime);
    }

    // Read validation data from FPGA will be placed sequentially in buffer for all replications
    // The data order should not matter, because every byte should have the same value!
    for (int r = 0; r < config.programSettings->kernelReplications; r++) {
        boRecv[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::copy(recvBufferContents[r].begin(), recvBufferContents[r].end(),
                  &validationData.data()[size_in_bytes * r]);
    }
    return network::ExecutionTimings{looplength, messageSize, calculationTimings};
}

} // namespace network::execution_types::udp

#endif
