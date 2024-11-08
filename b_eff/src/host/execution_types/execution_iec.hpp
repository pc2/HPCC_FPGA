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
#ifndef SRC_HOST_EXECUTION_TYPES_EXECUTION_IEC_HPP
#define SRC_HOST_EXECUTION_TYPES_EXECUTION_IEC_HPP

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "CL/cl_ext_intelfpga.h"
#include "mpi.h"

/* Project's headers */

namespace network::execution_types::iec {

    /*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
   template<class TDevice, class TContext, class TProgram>
    network::ExecutionTimings
    calculate(hpcc_base::ExecutionSettings<network::NetworkProgramSettings, TDevice, TContext, TProgram> const& config, cl_uint messageSize, cl_uint looplength,
                cl::vector<HOST_DATA_TYPE> &validationData) {

        int err;
        std::vector<cl::Kernel> sendKernels;
        std::vector<cl::Kernel> recvKernels;
        std::vector<cl::CommandQueue> sendQueues;
        std::vector<cl::CommandQueue> recvQueues;
        std::vector<cl::Buffer> validationBuffers;

        // Create all kernels and buffers. The kernel pairs are generated twice to utilize all channels
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {

            validationBuffers.push_back(cl::Buffer(*config.context, CL_MEM_WRITE_ONLY, sizeof(HOST_DATA_TYPE) * validationData.size(),0,&err));
            ASSERT_CL(err)

            cl::Kernel sendKernel(*config.program, (SEND_KERNEL_NAME + std::to_string(r)).c_str(), &err);
            ASSERT_CL(err)

            err = sendKernel.setArg(0, messageSize);
            ASSERT_CL(err)
            err = sendKernel.setArg(1, looplength);
            ASSERT_CL(err)

            cl::Kernel recvKernel(*config.program, (RECV_KERNEL_NAME + std::to_string(r)).c_str(), &err);
            ASSERT_CL(err)
            err = recvKernel.setArg(0, validationBuffers[r]);
            ASSERT_CL(err)
            err = recvKernel.setArg(1, messageSize);
            ASSERT_CL(err)
            err = recvKernel.setArg(2, looplength);
            ASSERT_CL(err)

            cl::CommandQueue sendQueue(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)
            cl::CommandQueue recvQueue(*config.context, *config.device, 0, &err);
            ASSERT_CL(err)

            recvQueues.push_back(recvQueue);
            sendQueues.push_back(sendQueue);
            sendKernels.push_back(sendKernel);
            recvKernels.push_back(recvKernel);

        }

        std::vector<double> calculationTimings;
        for (uint r =0; r < config.programSettings->numRepetitions; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            auto startCalculation = std::chrono::high_resolution_clock::now();
#ifdef HOST_EMULATION_REORDER
            std::cout << "Reordering kernel execution for Intel emulation!" << std::endl;
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                sendQueues[i].enqueueNDRangeKernel(sendKernels[i], cl::NullRange, cl::NDRange(1));
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Send Enqueued " << r << "," << i << std::endl;
                #endif
            }
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                sendQueues[i].finish();
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Send done " << r << "," << i << std::endl;
                #endif
            } 
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                recvQueues[i].enqueueNDRangeKernel(recvKernels[i], cl::NullRange, cl::NDRange(1));
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Recv Enqueued " << r << "," << i << std::endl;
                #endif
            }
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                recvQueues[i].finish();
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Recv done " << r << "," << i << std::endl;
                #endif
            }      
#else
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                sendQueues[i].enqueueNDRangeKernel(sendKernels[i], cl::NullRange, cl::NDRange(1));
                recvQueues[i].enqueueNDRangeKernel(recvKernels[i], cl::NullRange, cl::NDRange(1));
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Enqueued " << r << "," << i << std::endl;
                #endif
            }
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                sendQueues[i].finish();
                #ifndef NDEBUG
                        int current_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Send done " << r << "," << i << std::endl;
                #endif
                recvQueues[i].finish();
                #ifndef NDEBUG
                        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
                        std::cout << "Rank " << current_rank << ": Recv done " << r << "," << i << std::endl;
                #endif
            }
#endif
            auto endCalculation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());
#ifndef NDEBUG
        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);
        std::cout << "Rank " << current_rank << ": Done " << r << std::endl;
#endif
        }
        // Read validation data from FPGA will be placed sequentially in buffer for all replications
        // The data order should not matter, because every byte should have the same value!
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {
            err = recvQueues[r].enqueueReadBuffer(validationBuffers[r], CL_TRUE, 0, sizeof(HOST_DATA_TYPE) * (1 << messageSize), &validationData.data()[r * (1 << messageSize)]);
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