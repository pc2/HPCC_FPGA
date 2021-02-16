#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"

#include <chrono>
#include <thread>


#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define CHUNK (1 << REGISTER_BLOCK_LOG)


class LinpackKernelCommunicationTorusTest : public testing::Test {

public:
    std::unique_ptr<linpack::LinpackBenchmark> bm;
    cl::vector<HOST_DATA_TYPE> data;
    const unsigned numberOfChannels = 4;
    const std::string channelOutName = "kernel_output_ch";
    const std::string channelInName = "kernel_input_ch";

    virtual void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        bm->getExecutionSettings().programSettings->matrixSize = 4 * BLOCK_SIZE;
        setupExternalChannelFiles();
    }

    virtual void TearDown() override {
        bm = nullptr;
    }

    /**
     * @brief Setup the external channels files for the execution of a benchmark kernel
     * 
     */
    virtual void
    setupExternalChannelFiles() {
        for (int i=0; i < numberOfChannels; i++) {
            std::string fname = channelOutName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc);
            fs.close();
        }
    }

    cl::vector<HOST_DATA_TYPE>
    getDataForCurrentRank(int rank) {
        cl::vector<HOST_DATA_TYPE> output(2 * BLOCK_SIZE * 2 * BLOCK_SIZE);
        bm->getExecutionSettings().programSettings->matrixSize = 4 * BLOCK_SIZE;
        auto data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        int toruswidth = 2;
        int toruscol = rank % toruswidth;
        int torusrow = rank / toruswidth;
        for (int row = 0; row < 2; row++) {
            for ( int col = 0; col < 2; col++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for ( int i = 0; i < BLOCK_SIZE; i++) {
                        output[((row * BLOCK_SIZE) + j) * 2 * BLOCK_SIZE + col * BLOCK_SIZE + i] = data->A[(((row * toruswidth + torusrow) * BLOCK_SIZE) + j) * 4 * BLOCK_SIZE + (col* toruswidth + toruscol) * BLOCK_SIZE + i];
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Get the Data sent over an external channel.
     * 
     * @param channel_id Id of the external channel. It is assumed to be conntected in the order 0->Top,1->Right,2->Bottom,3->Left.
     *                  So channel 0 will be connected to the channel 2 of the FPGA above the current FPGA in the 2D Torus. 
     * @param output_channel Boolean, if true , the output channel is read otherwise the input channel
     * @return std::vector<HOST_DATA_TYPE> The data that is contained in the output file of the channel
     */
    std::vector<HOST_DATA_TYPE>
    getDataFromExternalChannel(uint channel_id, bool output_channel) {
        std::string fname = ( output_channel ? channelOutName : channelInName) + std::to_string(channel_id);
        std::ifstream fs;
        fs.open(fname, std::ifstream::binary | std::ifstream::in);
        HOST_DATA_TYPE value;
        std::vector<HOST_DATA_TYPE> values;
        while (fs.read(reinterpret_cast<char*>(&value), sizeof(HOST_DATA_TYPE))) {
            values.push_back(value);
        }
        return values;
    }

    cl::vector<HOST_DATA_TYPE>
    getResultMatrix(int rank) {
        cl::vector<HOST_DATA_TYPE> output(2 * BLOCK_SIZE * 2 * BLOCK_SIZE);
        bm->getExecutionSettings().programSettings->matrixSize = 4 * BLOCK_SIZE;
        auto data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        linpack::gefa_ref_nopvt(data->A,4 * BLOCK_SIZE,4 * BLOCK_SIZE);
        int toruswidth = 2;
        int toruscol = rank % 2;
        int torusrow = rank / 2;
        for (int row = 0; row < toruswidth; row++) {
            for ( int col = 0; col < toruswidth; col++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for ( int i = 0; i < BLOCK_SIZE; i++) {
                        output[((row * BLOCK_SIZE) + j) * 2 * BLOCK_SIZE + col * BLOCK_SIZE + i] = data->A[(((row * toruswidth + torusrow) * BLOCK_SIZE) + j) * 4 * BLOCK_SIZE + (col * toruswidth + toruscol) * BLOCK_SIZE + i];
                    }
                }
            }
        }
        return output;
    }

};

class LinpackKernelCommunicationTestTorus00 : public LinpackKernelCommunicationTorusTest {

    void SetUp() override {
        LinpackKernelCommunicationTorusTest::SetUp();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        data = getDataForCurrentRank(0);
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        for (int i=0; i < 3; i++) {
            // Fill all input channels with the correct number of 1.0s
            std::string fname = channelInName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            fs.close();
        }
        bm->getExecutionSettings().programSettings->matrixSize = 4 * BLOCK_SIZE;
        auto data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        linpack::gefa_ref_nopvt(data->A,4 * BLOCK_SIZE,4 * BLOCK_SIZE);
        // From bottom
        std::string fname = channelInName + std::to_string(1);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int j = 0; j < BLOCK_SIZE; j++) {
            for ( int i = 0; i < BLOCK_SIZE; i++) {
                fs.write(reinterpret_cast<const char*>(&data->A[(BLOCK_SIZE + j) * 4 * BLOCK_SIZE + 2 * BLOCK_SIZE + i]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
        // From right
        fname = channelInName + std::to_string(3);
        std::remove(fname.c_str());
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int j = 0; j < BLOCK_SIZE; j++) {
            for ( int i = 0; i < BLOCK_SIZE; i++) {
                fs.write(reinterpret_cast<const char*>(&data->A[(2 * BLOCK_SIZE + i) * 4 * BLOCK_SIZE + BLOCK_SIZE + j]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue left_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue top_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue inner_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer_left(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer lu_buffer_top(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer top_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer left_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);        
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel innerkernel(*bm->getExecutionSettings().program, "inner_update_mm0", &err);

        err = innerkernel.setArg(0, buffer);
        err = innerkernel.setArg(1, left_buffer_inner);
        err = innerkernel.setArg(2, top_buffer_inner);
        err = innerkernel.setArg(3, 1);
        err = innerkernel.setArg(4, 1);
        err = innerkernel.setArg(5, 2);

        cl::Kernel innerkernel2(*bm->getExecutionSettings().program, "inner_update_mm0", &err);

        err = innerkernel2.setArg(0, buffer);
        err = innerkernel2.setArg(1, left_buffer_inner);
        err = innerkernel2.setArg(2, top_buffer_inner);
        err = innerkernel2.setArg(3, 1);
        err = innerkernel2.setArg(4, 1);
        err = innerkernel2.setArg(5, 2);

        cl::Kernel leftkernel(*bm->getExecutionSettings().program, "left_update", &err);

        err = leftkernel.setArg(0, buffer);
        err = leftkernel.setArg(1, lu_buffer_left);
        err = leftkernel.setArg(2, CL_TRUE);
        err = leftkernel.setArg(3, 0);
        err = leftkernel.setArg(4, 1);
        err = leftkernel.setArg(5, 2);

        cl::Kernel topkernel(*bm->getExecutionSettings().program, "top_update", &err);

        err = topkernel.setArg(0, buffer);
        err = topkernel.setArg(1, lu_buffer_top);
        err = topkernel.setArg(2, CL_TRUE);
        err = topkernel.setArg(3, 1);
        err = topkernel.setArg(4, 0);
        err = topkernel.setArg(5, 2);

        cl::Kernel lu1kernel(*bm->getExecutionSettings().program, "lu", &err);

        err = lu1kernel.setArg(0, buffer);
        err = lu1kernel.setArg(1, 0);
        err = lu1kernel.setArg(2, 0);
        err = lu1kernel.setArg(3, 2);

        cl::Kernel lu2kernel(*bm->getExecutionSettings().program, "lu", &err);

        err = lu2kernel.setArg(0, buffer);
        err = lu2kernel.setArg(1, 1);
        err = lu2kernel.setArg(2, 1);
        err = lu2kernel.setArg(3, 2);

        // Start network layer kernel
        cl::Kernel network1(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network1.setArg(0, network_buffer);
        err = network1.setArg(1, left_buffer_inner);
        err = network1.setArg(2, top_buffer_inner);
        err = network1.setArg(3, static_cast<cl_uint>(STORE_TOP_INNER | STORE_LEFT_INNER | LEFT_BLOCK | TOP_BLOCK| LEFT_BLOCK_OUT | TOP_BLOCK_OUT | LU_BLOCK_OUT));
        err = network1.setArg(4, NETWORK_FWD_TOP | NETWORK_FWD_LEFT | NETWORK_FWD_RIGHT | NETWORK_FWD_BOTTOM);
        // Start network layer kernel
        cl::Kernel network2(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network2.setArg(0, network_buffer);
        err = network2.setArg(1, left_buffer_inner);
        err = network2.setArg(2, top_buffer_inner);
        err = network2.setArg(3, static_cast<cl_uint>(STORE_TOP_INNER | STORE_LEFT_INNER));
        err = network2.setArg(4, NETWORK_FWD_TOP | NETWORK_FWD_LEFT);
        cl::Kernel network3(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network3.setArg(0, network_buffer);
        err = network3.setArg(1, left_buffer_inner);
        err = network3.setArg(2, top_buffer_inner);
        err = network3.setArg(3, static_cast<cl_uint>(LU_BLOCK_OUT));
        err = network3.setArg(4, NETWORK_FWD_RIGHT | NETWORK_FWD_BOTTOM);
        network_queue.enqueueNDRangeKernel(network1, cl::NullRange, cl::NDRange(1),cl::NullRange);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data.data());
        compute_queue.enqueueNDRangeKernel(lu1kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        left_queue.enqueueNDRangeKernel(leftkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.enqueueNDRangeKernel(topkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        left_queue.finish();
        top_queue.finish();
        compute_queue.finish();
        inner_queue.enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();
        network_queue.enqueueNDRangeKernel(network2, cl::NullRange, cl::NDRange(1),cl::NullRange);
        network_queue.finish();
        inner_queue.enqueueNDRangeKernel(innerkernel2, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();
        network_queue.enqueueNDRangeKernel(network3, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.enqueueNDRangeKernel(lu2kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        network_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data.data());

    }
};


TEST_F(LinpackKernelCommunicationTestTorus00, AllBlockExternalResultisCorrect) {
    auto ref_data = getResultMatrix(0);

    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
            std::cout << std::abs(ref_data[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data[i * bm->getExecutionSettings().programSettings->matrixSize + j]) << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    EXPECT_FLOAT_EQ(total_error, 0.0);
}


class LinpackKernelCommunicationTestTorus01 : public LinpackKernelCommunicationTorusTest {

    void SetUp() override {
        LinpackKernelCommunicationTorusTest::SetUp();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        data = getDataForCurrentRank(1);
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        for (int i=0; i < 3; i++) {
            // Fill all input channels with the correct number of 1.0s
            std::string fname = channelInName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            fs.close();
        }
        bm->getExecutionSettings().programSettings->matrixSize = 4 * BLOCK_SIZE;
        auto data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        linpack::gefa_ref_nopvt(data->A,4 * BLOCK_SIZE,4 * BLOCK_SIZE);
        // From left
        std::string fname = channelInName + std::to_string(2);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = (ii / CHUNK) * CHUNK; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&data->A[jj * 4 * BLOCK_SIZE + ii]), sizeof(HOST_DATA_TYPE));
            }
        }
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = (ii / CHUNK) * CHUNK; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&data->A[(jj + 2 * BLOCK_SIZE) * 4 * BLOCK_SIZE + (ii + 2 * BLOCK_SIZE)]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
        // From top
        fname = channelInName + std::to_string(0);
        std::remove(fname.c_str());
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = (ii / CHUNK) * CHUNK; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&data->A[(ii + BLOCK_SIZE) * 4 * BLOCK_SIZE + jj + BLOCK_SIZE]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
        // From right
        fname = channelInName + std::to_string(3);
        std::remove(fname.c_str());
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = 0; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&data->A[(jj + 2 * BLOCK_SIZE) * 4 * BLOCK_SIZE + ii]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
        // From bottom
        fname = channelInName + std::to_string(1);
        std::remove(fname.c_str());
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = 0; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&data->A[(ii + BLOCK_SIZE) * 4 * BLOCK_SIZE + jj + 3 * BLOCK_SIZE]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue left_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue top_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue inner_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer_left(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer lu_buffer_top(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer top_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);
        cl::Buffer left_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);     
        cl::Buffer dummy_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE);     
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel innerkernel01(*bm->getExecutionSettings().program, "inner_update_mm0", &err);

        err = innerkernel01.setArg(0, buffer);
        err = innerkernel01.setArg(1, left_buffer_inner);
        err = innerkernel01.setArg(2, top_buffer_inner);
        err = innerkernel01.setArg(3, 0);
        err = innerkernel01.setArg(4, 1);
        err = innerkernel01.setArg(5, 2);

        cl::Kernel innerkernel11_1(*bm->getExecutionSettings().program, "inner_update_mm0", &err);

        err = innerkernel11_1.setArg(0, buffer);
        err = innerkernel11_1.setArg(1, left_buffer_inner);
        err = innerkernel11_1.setArg(2, top_buffer_inner);
        err = innerkernel11_1.setArg(3, 1);
        err = innerkernel11_1.setArg(4, 1);
        err = innerkernel11_1.setArg(5, 2);


        cl::Kernel topkernel00(*bm->getExecutionSettings().program, "top_update", &err);

        err = topkernel00.setArg(0, buffer);
        err = topkernel00.setArg(1, lu_buffer_top);
        err = topkernel00.setArg(2, CL_TRUE);
        err = topkernel00.setArg(3, 0);
        err = topkernel00.setArg(4, 0);
        err = topkernel00.setArg(5, 2);

        cl::Kernel topkernel10(*bm->getExecutionSettings().program, "top_update", &err);

        err = topkernel10.setArg(0, buffer);
        err = topkernel10.setArg(1, lu_buffer_top);
        err = topkernel10.setArg(2, CL_FALSE); //TODO loading from global memory seems to lead to error in calculation
        err = topkernel10.setArg(3, 1);
        err = topkernel10.setArg(4, 0);
        err = topkernel10.setArg(5, 2);


        cl::Kernel leftkernel01(*bm->getExecutionSettings().program, "left_update", &err);

        err = leftkernel01.setArg(0, buffer);
        err = leftkernel01.setArg(1, lu_buffer_left);
        err = leftkernel01.setArg(2, CL_TRUE);
        err = leftkernel01.setArg(3, 0);
        err = leftkernel01.setArg(4, 1);
        err = leftkernel01.setArg(5, 2);

        cl::Kernel innerkernel11_2(*bm->getExecutionSettings().program, "inner_update_mm0", &err);

        err = innerkernel11_2.setArg(0, buffer);
        err = innerkernel11_2.setArg(1, left_buffer_inner);
        err = innerkernel11_2.setArg(2, top_buffer_inner);
        err = innerkernel11_2.setArg(3, 1);
        err = innerkernel11_2.setArg(4, 1);
        err = innerkernel11_2.setArg(5, 2);

        cl::Kernel topkernel11(*bm->getExecutionSettings().program, "top_update", &err);

        err = topkernel11.setArg(0, buffer);
        err = topkernel11.setArg(1, lu_buffer_top);
        err = topkernel11.setArg(2, CL_TRUE);
        err = topkernel11.setArg(3, 1);
        err = topkernel11.setArg(4, 1);
        err = topkernel11.setArg(5, 2);

        // Start network layer kernel
        cl::Kernel network1(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network1.setArg(0, network_buffer);
        err = network1.setArg(1, left_buffer_inner);
        err = network1.setArg(2, top_buffer_inner);
        err = network1.setArg(3, static_cast<cl_uint>(STORE_TOP_INNER | STORE_LEFT_INNER | TOP_BLOCK| TOP_BLOCK_OUT));
        err = network1.setArg(4, 0);
        // Start network layer kernel
        cl::Kernel network2(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network2.setArg(0, network_buffer);
        err = network2.setArg(1, dummy_buffer_inner);
        err = network2.setArg(2, top_buffer_inner);
        err = network2.setArg(3, static_cast<cl_uint>(STORE_TOP_INNER | TOP_BLOCK_OUT));
        err = network2.setArg(4, 0);
        cl::Kernel network3(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network3.setArg(0, network_buffer);
        err = network3.setArg(1, left_buffer_inner);
        err = network3.setArg(2, top_buffer_inner);
        err = network3.setArg(3, static_cast<cl_uint>(STORE_LEFT_INNER | STORE_TOP_INNER | LEFT_BLOCK | LEFT_BLOCK_OUT));
        err = network3.setArg(4, 0);
        cl::Kernel network5(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network5.setArg(0, network_buffer);
        err = network5.setArg(1, left_buffer_inner);
        err = network5.setArg(2, top_buffer_inner);
        err = network5.setArg(3, static_cast<cl_uint>(TOP_BLOCK| TOP_BLOCK_OUT));
        err = network5.setArg(4, 0);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data.data());
        network_queue.enqueueNDRangeKernel(network1, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.enqueueNDRangeKernel(topkernel00, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.finish();
        network_queue.finish();

        inner_queue.enqueueNDRangeKernel(innerkernel01, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();

        top_queue.enqueueNDRangeKernel(topkernel10, cl::NullRange, cl::NDRange(1),cl::NullRange);
        network_queue.enqueueNDRangeKernel(network2, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.finish();
        network_queue.finish();

        inner_queue.enqueueNDRangeKernel(innerkernel11_1, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();

        network_queue.enqueueNDRangeKernel(network3, cl::NullRange, cl::NDRange(1),cl::NullRange);
        left_queue.enqueueNDRangeKernel(leftkernel01, cl::NullRange, cl::NDRange(1),cl::NullRange);
        left_queue.finish();
        network_queue.finish();

        inner_queue.enqueueNDRangeKernel(innerkernel11_2, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();

        network_queue.enqueueNDRangeKernel(network5, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.enqueueNDRangeKernel(topkernel11, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.finish();
        network_queue.finish();
        
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data.data());

    }
};


TEST_F(LinpackKernelCommunicationTestTorus01, AllBlockExternalResultisCorrect) {
    auto ref_data = getResultMatrix(1);

    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
            std::cout << std::abs(ref_data[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data[i * bm->getExecutionSettings().programSettings->matrixSize + j]) << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    EXPECT_FLOAT_EQ(total_error, 0.0);
}


