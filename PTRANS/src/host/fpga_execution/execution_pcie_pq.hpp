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
#ifndef SRC_HOST_PCIE_PQ_EXECUTION_H_
#define SRC_HOST_PCIE_PQ_EXECUTION_H_

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
namespace pcie_pq {

    /**
 * @brief Transpose and add the matrices using the OpenCL kernel using a PQ distribution and PCIe+MPI over the host for communication
 * 
 * @param config The progrma configuration
 * @param data data object that contains all required data for the execution on the FPGA
 * @param handler data handler instance that should be used to exchange data between hosts
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured execution times 
 */
static  std::unique_ptr<transpose::TransposeExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings>& config, transpose::TransposeData& data, transpose::data_handler::TransposeDataHandler &handler) {
        int err;

        if (config.programSettings->dataHandlerIdentifier != transpose::data_handler::DataHandlerType::pq) {
                throw std::runtime_error("Used data handler not supported by execution handler!");
        }
#ifdef USE_SVM
        throw new std::runtime_error("SVM not supported in the host implementation of this communication method");
#endif

        std::vector<size_t> bufferSizeList;
        std::vector<size_t> bufferStartList;
        std::vector<size_t> bufferOffsetList;
        std::vector<cl::Buffer> bufferListA;
        std::vector<cl::Buffer> bufferListB;
        std::vector<cl::Buffer> bufferListA_out;
        std::vector<cl::Kernel> transposeKernelList;
        std::vector<cl::CommandQueue> transCommandQueueList;

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
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
                cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY | memory_bank_info_a,
                                buffer_size * sizeof(HOST_DATA_TYPE));
#else
                cl::Buffer bufferA(*config.context, CL_MEM_READ_ONLY | memory_bank_info_a,
                                data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE));
#endif
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
                err = transposeKernel.setArg(4, static_cast<cl_uint>(blocks_per_replication));
                ASSERT_CL(err)
                err = transposeKernel.setArg(5, static_cast<cl_uint>(local_matrix_width));
                ASSERT_CL(err)
#ifndef USE_BUFFER_WRITE_RECT_FOR_A
                err = transposeKernel.setArg(6, static_cast<cl_uint>(local_matrix_width));
                ASSERT_CL(err) 
                err = transposeKernel.setArg(3, static_cast<cl_uint>(bufferStartList[r]));
                ASSERT_CL(err)
#else
                err = transposeKernel.setArg(6, static_cast<cl_uint>((bufferSizeList[r]) / (local_matrix_width * data.blockSize * data.blockSize)));
                ASSERT_CL(err) 
                err = transposeKernel.setArg(3, static_cast<cl_uint>(0));
                ASSERT_CL(err)
#endif
 

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

        for (int repetition = 0; repetition < config.programSettings->numRepetitions; repetition++) {

            auto startTransfer = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < transposeKernelList.size(); r++) {
                transCommandQueueList[r].enqueueWriteBuffer(bufferListB[r], CL_FALSE, 0,
                                        bufferSizeList[r]* sizeof(HOST_DATA_TYPE), &data.B[bufferStartList[r] * data.blockSize * data.blockSize]);
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
                cl::size_t<3> deviceOffset;
                cl::size_t<3> hostOffset;
                hostOffset[0] = (bufferStartList[r]) / (local_matrix_width) * data.blockSize * sizeof(HOST_DATA_TYPE);
                cl::size_t<3> rectShape;
                rectShape[0] = (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE);
                rectShape[1] = local_matrix_width* data.blockSize;
                rectShape[2] = 1L;
                transCommandQueueList[r].enqueueWriteBufferRect(bufferListA[r],CL_FALSE, 
                                                deviceOffset, 
                                                hostOffset, 
                                                rectShape,
                                                (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE), 0,
                                                local_matrix_width* data.blockSize*sizeof(HOST_DATA_TYPE), 0,
                                                data.A);
#else
                transCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_FALSE, 0,
                                        data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE), data.A);
#endif

        }
            for (int r = 0; r < transposeKernelList.size(); r++) {
                transCommandQueueList[r].finish();
            }
            auto endTransfer = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> transferTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);

            MPI_Barrier(MPI_COMM_WORLD);

            auto startCalculation = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < transposeKernelList.size(); r++)
        {
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
                cl::size_t<3> deviceOffset;
                cl::size_t<3> hostOffset;
                hostOffset[0] = (bufferStartList[r]) / (local_matrix_width) * data.blockSize * sizeof(HOST_DATA_TYPE);
                cl::size_t<3> rectShape;
                rectShape[0] = (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE);
                rectShape[1] = local_matrix_width* data.blockSize;
                rectShape[2] = 1L;
                transCommandQueueList[r].enqueueReadBufferRect(bufferListA[r],CL_FALSE, 
                                                deviceOffset, 
                                                hostOffset, 
                                                rectShape,
                                                (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE), 0,
                                                local_matrix_width* data.blockSize*sizeof(HOST_DATA_TYPE), 0,
                                                data.A);
#else
                transCommandQueueList[r].enqueueReadBuffer(bufferListA[r], CL_FALSE, 0,
                                        data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE), data.A);
#endif
        }

        // Exchange A data via PCIe and MPI
        handler.exchangeData(data);

        for (int r = 0; r < transposeKernelList.size(); r++)
        {
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
                cl::size_t<3> deviceOffset;
                cl::size_t<3> hostOffset;
                hostOffset[0] = (bufferStartList[r]) / (local_matrix_width) * data.blockSize * sizeof(HOST_DATA_TYPE);
                cl::size_t<3> rectShape;
                rectShape[0] = (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE);
                rectShape[1] = local_matrix_width* data.blockSize;
                rectShape[2] = 1L;
                transCommandQueueList[r].enqueueWriteBufferRect(bufferListA[r],CL_FALSE, 
                                                deviceOffset, 
                                                hostOffset, 
                                                rectShape,
                                                (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE), 0,
                                                local_matrix_width* data.blockSize*sizeof(HOST_DATA_TYPE), 0,
                                                data.A);
#else
                transCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_FALSE, 0,
                                        data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE), data.A);
#endif
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
                std::cout << "Rank " << mpi_rank << ": " << "Done i=" << repetition << std::endl;
#endif

        // Transfer back data for next repetition!
        handler.exchangeData(data);

            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());

            startTransfer = std::chrono::high_resolution_clock::now();

                for (int r = 0; r < transposeKernelList.size(); r++) {
                        transCommandQueueList[r].enqueueReadBuffer(bufferListA_out[r], CL_TRUE, 0,
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