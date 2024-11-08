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

/* Project's headers */
#include "transpose_benchmark.hpp"
#include "data_handlers/data_handler_types.h"
#include "data_handlers/pq.hpp"

namespace transpose {
namespace fpga_execution {
namespace intel_pq {

    /**
 * @brief Transpose and add the matrices using the OpenCL kernel using a PQ distribution and Intel external channels for communication
 * 
 * @param config The progrma configuration
 * @param data data object that contains all required data for the execution on the FPGA
 * @return std::unique_ptr<transpose::TransposeExecutionTimings> The measured execution times 
 */
    static std::map<std::string, std::vector<double>>
    calculate(const hpcc_base::ExecutionSettings<transpose::TransposeProgramSettings, cl::Device, cl::Context, cl::Program>& config, transpose::TransposeData<cl::Context>& data, transpose::data_handler::DistributedPQTransposeDataHandler<cl::Device, cl::Context, cl::Program> &handler) {
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
        std::vector<cl::Kernel> transposeReadKernelList;
        std::vector<cl::Kernel> transposeWriteKernelList;
        std::vector<cl::CommandQueue> readCommandQueueList;
        std::vector<cl::CommandQueue> writeCommandQueueList;

        size_t local_matrix_width = handler.getWidthforRank();
        size_t local_matrix_height = handler.getHeightforRank();
        size_t local_matrix_width_bytes = local_matrix_width * data.blockSize * sizeof(HOST_DATA_TYPE);

        size_t total_offset = 0;
        size_t row_offset = 0;

        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


        if (config.programSettings->p * config.programSettings->p != mpi_size) {
                throw std::runtime_error("P=Q must hold for IEC implementation, but P=" + std::to_string(config.programSettings->p) + " and Q=" + std::to_string(mpi_size / config.programSettings->p));
        }

        // Setup the kernels depending on the number of kernel replications
        for (int r = 0; r < config.programSettings->kernelReplications; r++) {

                // Calculate how many blocks the current kernel replication will need to process.
                size_t blocks_per_replication = (local_matrix_height * local_matrix_width / config.programSettings->kernelReplications);
                size_t blocks_remainder = (local_matrix_height * local_matrix_width) % config.programSettings->kernelReplications;
                if (blocks_remainder > r) {
                        // Catch the case, that the number of blocks is not divisible by the number of kernel replications
                        blocks_per_replication += 1;
                }
                if (blocks_per_replication < 1) {
                        continue;
                }

                size_t buffer_size = (blocks_per_replication + local_matrix_width - 1) / local_matrix_width * local_matrix_width * data.blockSize * data.blockSize;
                bufferSizeList.push_back(buffer_size);
                bufferStartList.push_back(total_offset);
                bufferOffsetList.push_back(row_offset);

                row_offset = (row_offset + blocks_per_replication) % local_matrix_width;

                total_offset += (bufferOffsetList.back() + blocks_per_replication) / local_matrix_width * local_matrix_width;

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
                err = transposeWriteKernel.setArg(2, static_cast<cl_ulong>(bufferOffsetList[r]));
                ASSERT_CL(err)
        
                // Width of the whole local matrix in blocks
                err = transposeWriteKernel.setArg(3, static_cast<cl_ulong>(local_matrix_width));
                ASSERT_CL(err) 
#ifndef USE_BUFFER_WRITE_RECT_FOR_A
                // Row offset in blocks
                err = transposeReadKernel.setArg(1, static_cast<cl_ulong>(bufferStartList[r] + bufferOffsetList[r]));
                ASSERT_CL(err)   
                err = transposeReadKernel.setArg(2, static_cast<cl_ulong>(local_matrix_width));
                ASSERT_CL(err) 
#else
                // Row offset in blocks
                err = transposeReadKernel.setArg(1, static_cast<cl_ulong>(0));
                ASSERT_CL(err) 
                err = transposeReadKernel.setArg(2, static_cast<cl_ulong>((bufferSizeList[r]) / (local_matrix_width * data.blockSize * data.blockSize)));
                ASSERT_CL(err) 
#endif

                // Height of the whole local matrix in blocks
                err = transposeReadKernel.setArg(3, static_cast<cl_ulong>(local_matrix_height ));
                ASSERT_CL(err) 

                // total number of blocks that are processed in this replication
                err = transposeWriteKernel.setArg(4, static_cast<cl_ulong>(blocks_per_replication));
                ASSERT_CL(err) 
                err = transposeReadKernel.setArg(4, static_cast<cl_ulong>(blocks_per_replication));
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
#ifdef USE_BUFFER_WRITE_RECT_FOR_A
#ifndef USE_DEPRECATED_HPP_HEADER
                cl::array<size_t,3> deviceOffset;
                cl::array<size_t,3> hostOffset;
                cl::array<size_t,3> rectShape;
#else
                cl::size_t<3> deviceOffset;
                cl::size_t<3> hostOffset;
                cl::size_t<3> rectShape;
#endif
                deviceOffset[0] = 0;
                deviceOffset[1] = 0;
                deviceOffset[2] = 0;
                hostOffset[0] = (bufferStartList[r]) / local_matrix_width * data.blockSize * sizeof(HOST_DATA_TYPE);
                hostOffset[1] = 0;
                hostOffset[2] = 0;
                rectShape[0] = (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE);
                rectShape[1] = local_matrix_width* data.blockSize;
                rectShape[2] = 1L;
                readCommandQueueList[r].enqueueWriteBufferRect(bufferListA[r],CL_FALSE, 
                                                deviceOffset, 
                                                hostOffset, 
                                                rectShape,
                                                (bufferSizeList[r]) / (local_matrix_width * data.blockSize) * sizeof(HOST_DATA_TYPE), 0,
                                                local_matrix_width* data.blockSize*sizeof(HOST_DATA_TYPE), 0,
                                                data.A);
#else
                readCommandQueueList[r].enqueueWriteBuffer(bufferListA[r], CL_FALSE, 0,
                                        data.numBlocks * data.blockSize * data.blockSize * sizeof(HOST_DATA_TYPE), data.A);
#endif

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
#ifdef HOST_EMULATION_REORDER
        std::cout << "Reorder kernel execution on host for Intel fast emulation!" << std::endl;
        // We have to use the same kernel_output_ch files for the different MPI ranks, that means we have to
        // sequentialize the execution such that only a single rank sends something to the channels.
        for (int k = 0; k < mpi_size; k++) {
                int receiver_rank = 2 * (k%2) + (k/2);
                // If current rank, start sending to the channels
                if (k == mpi_rank) {
                        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                                readCommandQueueList[r].enqueueNDRangeKernel(transposeReadKernelList[r], cl::NullRange, cl::NDRange(1));
                        }
                        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                                readCommandQueueList[r].finish();
#ifndef NDEBUG
                                std::cout << "Rank " << mpi_rank << ": " << "Read done r=" << r << ", i=" << repetition << std::endl;
#endif
                        }
                }
                // Wait for all ranks to start receiving
                MPI_Barrier(MPI_COMM_WORLD);
                // Only rank that has to receive the data starts receive kernel
                if (receiver_rank == mpi_rank) {
                        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                                writeCommandQueueList[r].enqueueNDRangeKernel(transposeWriteKernelList[r], cl::NullRange, cl::NDRange(1));
                        }
                        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                                writeCommandQueueList[r].finish();
#ifndef NDEBUG
                                std::cout << "Rank " << mpi_rank << ": " << "Write done r=" << r << ", i=" << repetition << std::endl;
#endif
                        }
                        // Delete the channels files so next iteration starts with clean channels!
                        for (int r = 0; r < transposeReadKernelList.size(); r++) {
                                std::remove(("kernel_output_ch" + std::to_string(r)).c_str());
                        }
                }
                // Wait until receiving kernel is done
                MPI_Barrier(MPI_COMM_WORLD);
        }
#else
            for (int r = 0; r < transposeReadKernelList.size(); r++) {
                writeCommandQueueList[r].enqueueNDRangeKernel(transposeWriteKernelList[r], cl::NullRange, cl::NDRange(1));
                readCommandQueueList[r].enqueueNDRangeKernel(transposeReadKernelList[r], cl::NullRange, cl::NDRange(1));
            }
            for (int r = 0; r < transposeReadKernelList.size(); r++) {
                writeCommandQueueList[r].finish();
#ifndef NDEBUG
                std::cout << "Rank " << mpi_rank << ": " << "Write done r=" << r << ", i=" << repetition << std::endl;
#endif
                readCommandQueueList[r].finish();
#ifndef NDEBUG
                std::cout << "Rank " << mpi_rank << ": " << "Read done r=" << r << ", i=" << repetition << std::endl;
#endif
            }
#endif
            auto endCalculation = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
                std::cout << "Rank " << mpi_rank << ": " << "Done i=" << repetition << std::endl;
#endif
            std::chrono::duration<double> calculationTime =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endCalculation - startCalculation);
            calculationTimings.push_back(calculationTime.count());

            std::vector<HOST_DATA_TYPE> tmp_write_buffer(local_matrix_height * local_matrix_width * data.blockSize * data.blockSize); 

            startTransfer = std::chrono::high_resolution_clock::now();

                for (int r = 0; r < transposeReadKernelList.size(); r++) {
                        // Copy possibly incomplete first block row
                        if (bufferOffsetList[r] != 0) {
                                writeCommandQueueList[r].enqueueReadBuffer(bufferListA_out[r], CL_TRUE, 0,
                                                bufferSizeList[r]* sizeof(HOST_DATA_TYPE), tmp_write_buffer.data());
                                writeCommandQueueList[r].finish();
                                for (int row = 0; row < data.blockSize; row++) {
                                        for (int col = bufferOffsetList[r] * data.blockSize; col < local_matrix_width * data.blockSize; col++) {
                                                data.result[bufferStartList[r] * data.blockSize * data.blockSize + row * local_matrix_width * data.blockSize + col] =
                                                        tmp_write_buffer[row * local_matrix_width * data.blockSize + col];
                                        }
                                }
                                // Copy remaining buffer
                                std::copy(tmp_write_buffer.begin() + local_matrix_width * data.blockSize * data.blockSize, tmp_write_buffer.begin() + bufferSizeList[r],&data.result[(bufferStartList[r] + local_matrix_width) * data.blockSize * data.blockSize]);
                        }
                        else {
                                writeCommandQueueList[r].enqueueReadBuffer(bufferListA_out[r], CL_TRUE, 0,
                                                         bufferSizeList[r]* sizeof(HOST_DATA_TYPE), &data.result[bufferStartList[r] * data.blockSize * data.blockSize]);
                                writeCommandQueueList[r].finish();
                        }
                }
            endTransfer = std::chrono::high_resolution_clock::now();
            transferTime +=
                    std::chrono::duration_cast<std::chrono::duration<double>>
                            (endTransfer - startTransfer);
            transferTimings.push_back(transferTime.count());
        }

        std::map<std::string, std::vector<double>> timings;
        timings["transfer"] = transferTimings;
        timings["calculation"] = calculationTimings;
        return timings;
    }

}  // namespace transpose
}  // namespace fpga_execution
}  // namespace intel

#endif
