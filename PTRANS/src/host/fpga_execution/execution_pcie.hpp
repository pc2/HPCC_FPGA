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
#ifndef SRC_HOST_PCIE_EXECUTION_H_
#define SRC_HOST_PCIE_EXECUTION_H_

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "CL/cl.hpp"
#include "mpi.h"

/* Project's headers */
#include "data_handlers/handler.hpp"

namespace transpose
{
    namespace fpga_execution
    {
        namespace pcie
        {

            /**
 * @brief Transpose and add the matrices using the OpenCL kernel
 * 
 * @param config The progrma configuration
 * @param data data object that contains all required data for the execution on the FPGA
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured execution times 
 */
            static std::unique_ptr<transpose::TransposeExecutionTimings>
            calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings> &config, transpose::TransposeData &data, transpose::data_handler::TransposeDataHandler &handler)
            {
                int err;

#ifdef USE_SVM
                throw new std::runtime_error("SVM not supported in the host implementation of this communication method");
#endif

                std::vector<size_t> bufferSizeList;
                std::vector<cl::Buffer> bufferListA;
                std::vector<cl::Buffer> bufferListB;
                std::vector<cl::Buffer> bufferListA_out;
                std::vector<cl::Kernel> transposeKernelList;
                std::vector<cl::CommandQueue> transCommandQueueList;

                // Setup the kernels depending on the number of kernel replications
                for (int r = 0; r < config.programSettings->kernelReplications; r++)
                {

                    // Calculate how many blocks the current kernel replication will need to process.
                    size_t blocks_per_replication = data.numBlocks / config.programSettings->kernelReplications;
                    size_t blocks_remainder = data.numBlocks % config.programSettings->kernelReplications;
                    if (blocks_remainder > r)
                    {
                        // Catch the case, that the number of blocks is not divisible by the number of kernel replications
                        blocks_per_replication += 1;
                    }
                    if (blocks_per_replication < 1)
                    {
                        continue;
                    }

                    size_t buffer_size = data.blockSize * (data.blockSize * blocks_per_replication);

                    bufferSizeList.push_back(buffer_size);

                    int memory_bank_info_a = 0;
                    int memory_bank_info_b = 0;
                    int memory_bank_info_out = 0;
#ifdef INTEL_FPGA
                    if (!config.programSettings->useMemoryInterleaving)
                    {
                        // Define the memory bank the buffers will be placed in
                        if (config.programSettings->distributeBuffers)
                        {
                            memory_bank_info_a = ((((r * 3) % 7) + 1) << 16);
                            memory_bank_info_b = ((((r * 3 + 1) % 7) + 1) << 16);
                            memory_bank_info_out = ((((r * 3 + 2) % 7) + 1) << 16);
                        }
                        else
                        {
                            memory_bank_info_a = ((r + 1) << 16);
                            memory_bank_info_b = ((r + 1) << 16);
                            memory_bank_info_out = ((r + 1) << 16);
                        }
                    }
#endif
                    cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY | memory_bank_info_a,
                               buffer_size * sizeof(HOST_DATA_TYPE));
                    cl::Buffer bufferB(*config.context, CL_MEM_READ_ONLY | memory_bank_info_b,
                               buffer_size * sizeof(HOST_DATA_TYPE));
                    cl::Buffer bufferA_out(*config.context, CL_MEM_WRITE_ONLY | memory_bank_info_out,
                                   buffer_size * sizeof(HOST_DATA_TYPE));

                    // TODO the kernel name may need to be changed for Xilinx support
                    cl::Kernel transposeKernel(*config.program, ("transpose" + std::to_string(r)).c_str(), &err);
                    ASSERT_CL(err)


                    err = transposeKernel.setArg(0, bufferA);
                    ASSERT_CL(err)
                    err = transposeKernel.setArg(1, bufferB);
                    ASSERT_CL(err)
                    err = transposeKernel.setArg(2, bufferA_out);
                    ASSERT_CL(err)
                    err = transposeKernel.setArg(3, static_cast<cl_ulong>(0));
                    ASSERT_CL(err)
                    err = transposeKernel.setArg(4, static_cast<cl_ulong>(blocks_per_replication));
                    ASSERT_CL(err)

                    cl::CommandQueue transQueue(*config.context, *config.device, 0, &err);
                    ASSERT_CL(err)

                    transCommandQueueList.push_back(transQueue);
                    bufferListA.push_back(bufferA);
                    bufferListB.push_back(bufferB);
                    bufferListA_out.push_back(bufferA_out);
                    transposeKernelList.push_back(transposeKernel);
                }

                std::vector<double> transferTimings;
                std::vector<double> calculationTimings;

                for (int repetition = 0; repetition < config.programSettings->numRepetitions; repetition++)
                {

                    MPI_Barrier(MPI_COMM_WORLD);

                    auto startTransfer = std::chrono::high_resolution_clock::now();
                    size_t bufferOffset = 0;

                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].enqueueWriteBuffer(bufferListB[r], CL_TRUE, 0,
                                              bufferSizeList[r] * sizeof(HOST_DATA_TYPE), &data.B[bufferOffset]);
                        transCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_TRUE, 0,
                                              bufferSizeList[r] * sizeof(HOST_DATA_TYPE), &data.A[bufferOffset]);
                        bufferOffset += bufferSizeList[r];
                    }

                    auto endTransfer = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> transferTime =
                        std::chrono::duration_cast<std::chrono::duration<double>>(endTransfer - startTransfer);

                    MPI_Barrier(MPI_COMM_WORLD);

                    auto startCalculation = std::chrono::high_resolution_clock::now();
                    bufferOffset = 0;
                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].enqueueReadBuffer(bufferListA[r], CL_TRUE, 0,
                                               bufferSizeList[r] * sizeof(HOST_DATA_TYPE), &data.A[bufferOffset]);
                        bufferOffset += bufferSizeList[r];
                    }

                    // Exchange A data via PCIe and MPI
                    handler.exchangeData(data);

                    bufferOffset = 0;
                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_FALSE, 0,
                                                bufferSizeList[r] * sizeof(HOST_DATA_TYPE), &data.A[bufferOffset]);
                        bufferOffset += bufferSizeList[r];
                    }

                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].enqueueTask(transposeKernelList[r]);
                    }
                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].finish();
                    }

                    auto endCalculation = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
                    int mpi_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                    std::cout << "Rank " << mpi_rank << ": "
                          << "Done i=" << repetition << std::endl;
#endif
                    std::chrono::duration<double> calculationTime =
                        std::chrono::duration_cast<std::chrono::duration<double>>(endCalculation - startCalculation);
                    calculationTimings.push_back(calculationTime.count());

                    // Transfer back data for next repetition!
                    handler.exchangeData(data);

                    bufferOffset = 0;
                    startTransfer = std::chrono::high_resolution_clock::now();

                    for (int r = 0; r < transposeKernelList.size(); r++)
                    {
                        transCommandQueueList[r].enqueueReadBuffer(bufferListA_out[r], CL_TRUE, 0,
                                               bufferSizeList[r] * sizeof(HOST_DATA_TYPE), &data.result[bufferOffset]);
                        bufferOffset += bufferSizeList[r];
                    }

                    endTransfer = std::chrono::high_resolution_clock::now();
                    transferTime +=
                        std::chrono::duration_cast<std::chrono::duration<double>>(endTransfer - startTransfer);
                    transferTimings.push_back(transferTime.count());
                }

                std::unique_ptr<transpose::TransposeExecutionTimings> result(new transpose::TransposeExecutionTimings{
                    transferTimings,
                    calculationTimings});
                return result;
            }

        } // namespace bm_execution
    }
}

#endif // SRC_HOST_PCIE_EXECUTION_H_
