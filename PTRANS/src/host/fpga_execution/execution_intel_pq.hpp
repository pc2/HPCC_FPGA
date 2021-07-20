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
#ifndef SRC_HOST_INTEL_PQ_EXECUTION_H_
#define SRC_HOST_INTEL_PQ_EXECUTION_H_

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "CL/cl.hpp"

/* Project's headers */
#include "data_handlers/data_handler_types.h"

namespace transpose {
namespace fpga_execution {
namespace intel_pq {

    /**
 * @brief Transpose and add the matrices using the OpenCL kernel
 * 
 * @param config The progrma configuration
 * @param data data object that contains all required data for the execution on the FPGA
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured execution times 
 */
static  std::unique_ptr<transpose::TransposeExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& config, transpose::TransposeData& data) {
        int err;

        std::vector<size_t> bufferSizeList;
        std::vector<size_t> bufferStartList;
        std::vector<size_t> bufferOffsetList;
        std::vector<cl::Buffer> bufferListA;
        std::vector<cl::Buffer> bufferListB;
        std::vector<cl::Buffer> bufferListA_out;
        std::vector<cl::Kernel> transposeReadKernelList;
        std::vector<cl::Kernel> transposeWriteKernelList;
        std::vector<cl::CommandQueue> readCommandQueueList;
        std::vector<cl::CommandQueue> writeCommandQueueList;

        size_t local_matrix_width = std::sqrt(data.numBlocks);
        size_t local_matrix_width_bytes = local_matrix_width * data.blockSize * sizeof(HOST_DATA_TYPE);

        size_t total_offset = 0;

        // Setup the kernels depending on the number of kernel replications
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {

                // Calculate how many blocks the current kernel replication will need to process.
                size_t blocks_per_replication = (local_matrix_width / config.programSettings->kernelReplications * local_matrix_width);
                size_t blocks_remainder = local_matrix_width % config.programSettings->kernelReplications;
                if (blocks_remainder > r) {
                        // Catch the case, that the number of blocks is not divisible by the number of kernel replications
                        blocks_per_replication += local_matrix_width;
                }
                if (blocks_per_replication < 1) {
                        continue;
                }

                size_t buffer_size = blocks_per_replication * data.blockSize * data.blockSize;
                bufferSizeList.push_back(buffer_size);
                bufferStartList.push_back(total_offset);

                total_offset += blocks_per_replication;

                int memory_bank_info_a = 0;
                int memory_bank_info_b = 0;
                int memory_bank_info_out = 0;
#ifdef INTEL_FPGA
                if (!config.programSettings->useMemoryInterleaving) {
                        // Define the memory bank the buffers will be placed in
                        if (config.programSettings->distributeBuffers) {
                                memory_bank_info_a = ((((r * 3) % 7) + 1) << 16);
                                memory_bank_info_b = ((((r * 3 + 1) % 7) + 1) << 16);
                                memory_bank_info_out = ((((r * 3 + 2) % 7) + 1) << 16);
                        }
                        else {
                                memory_bank_info_a = ((r + 1) << 16);
                                memory_bank_info_b = ((r + 1) << 16);
                                memory_bank_info_out = ((r + 1) << 16);
                        }
                }
#endif
                cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY | memory_bank_info_a,
                                data.numBlocks * data.blockSize * data.blockSize* sizeof(HOST_DATA_TYPE));
                cl::Buffer bufferB(*config.context, CL_MEM_READ_ONLY | memory_bank_info_b,
                                buffer_size * sizeof(HOST_DATA_TYPE));
                cl::Buffer bufferA_out(*config.context, CL_MEM_WRITE_ONLY | memory_bank_info_out,
                                buffer_size * sizeof(HOST_DATA_TYPE));

                // TODO the kernel name may need to be changed for Xilinx support
                cl::Kernel transposeReadKernel(*config.program, (READ_KERNEL_NAME + std::to_string(r)).c_str(), &err);
                ASSERT_CL(err)
                cl::Kernel transposeWriteKernel(*config.program, (WRITE_KERNEL_NAME + std::to_string(r)).c_str(), &err);
                ASSERT_CL(err)

                err = transposeReadKernel.setArg(0, bufferA);
                ASSERT_CL(err)   
                err = transposeWriteKernel.setArg(0, bufferB);
                ASSERT_CL(err)
                err = transposeWriteKernel.setArg(1, bufferA_out);
                ASSERT_CL(err)

                // Row offset in blocks
                err = transposeReadKernel.setArg(1, static_cast<cl_ulong>(bufferStartList[r]));
                ASSERT_CL(err)   
                err = transposeWriteKernel.setArg(2, static_cast<cl_ulong>(0));
                ASSERT_CL(err)
        
                // Width or heigth of the whole local matrix in blocks
                err = transposeWriteKernel.setArg(3, static_cast<cl_ulong>(local_matrix_width));
                ASSERT_CL(err) 
                err = transposeReadKernel.setArg(2, static_cast<cl_ulong>(local_matrix_width));
                ASSERT_CL(err) 

                // total number of blocks that are processed in this replication
                err = transposeWriteKernel.setArg(4, static_cast<cl_ulong>(blocks_per_replication));
                ASSERT_CL(err) 
                err = transposeReadKernel.setArg(3, static_cast<cl_ulong>(blocks_per_replication));
                ASSERT_CL(err)     

                cl::CommandQueue readQueue(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)
                cl::CommandQueue writeQueue(*config.context, *config.device, 0, &err);
                ASSERT_CL(err)

                readCommandQueueList.push_back(readQueue);
                writeCommandQueueList.push_back(writeQueue);
                bufferListA.push_back(bufferA);
                bufferListB.push_back(bufferB);
                bufferListA_out.push_back(bufferA_out);
                transposeReadKernelList.push_back(transposeReadKernel);
                transposeWriteKernelList.push_back(transposeWriteKernel);
        }

        std::vector<double> transferTimings;
        std::vector<double> calculationTimings;

        for (int repetition = 0; repetition < config.programSettings->numRepetitions; repetition++) {

            auto startTransfer = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                writeCommandQueueList[r].enqueueWriteBuffer(bufferListB[r], CL_FALSE, 0,
                                        bufferSizeList[r]* sizeof(HOST_DATA_TYPE), &data.B[bufferStartList[r] * data.blockSize * data.blockSize]);
                // TODO: The dull local buffer of A is written to each bank since data is read columnwise. Use write buffer rect to simplify this?
                readCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_FALSE, 0,
                                        data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE), data.A);

        }
            for (int r = 0; r < transposeReadKernelList.size(); r++) {
                readCommandQueueList[r].finish();
                writeCommandQueueList[r].finish();
            }
            auto endTransfer = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> transferTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);

            MPI_Barrier(MPI_COMM_WORLD);

            auto startCalculation = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < transposeReadKernelList.size(); r++) {
                writeCommandQueueList[r].enqueueTask(transposeWriteKernelList[r]);
                readCommandQueueList[r].enqueueTask(transposeReadKernelList[r]);
            }
            for (int r = 0; r < transposeReadKernelList.size(); r++) {
                writeCommandQueueList[r].finish();
                readCommandQueueList[r].finish();
            }
            auto endCalculation = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
                int mpi_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                std::cout << "Rank " << mpi_rank << ": " << "Done i=" << repetition << std::endl;
#endif
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());

            startTransfer = std::chrono::high_resolution_clock::now();

                for (int r = 0; r < transposeReadKernelList.size(); r++) {
                        writeCommandQueueList[r].enqueueReadBuffer(bufferListA_out[r], CL_TRUE, 0,
                                                bufferSizeList[r]* sizeof(HOST_DATA_TYPE), &data.result[bufferStartList[r] * data.blockSize * data.blockSize]);
                }
            endTransfer = std::chrono::high_resolution_clock::now();
            transferTime +=
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);
            transferTimings.push_back(transferTime.count());
        }

        std::unique_ptr<transpose::TransposeExecutionTimings> result(new transpose::TransposeExecutionTimings{
                transferTimings,
                calculationTimings
        });
        return result;
    }

}  // namespace transpose
}  // namespace fpga_execution
}  // namespace intel

#endif