#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"

#include <chrono>
#include <thread>


#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define CHUNK (1 << REGISTER_BLOCK_LOG)


class LinpackKernelCommunicationTest : public testing::Test {

public:
    std::unique_ptr<linpack::LinpackBenchmark> bm;
    std::unique_ptr<linpack::LinpackData> data;
    const unsigned numberOfChannels = 4;
    const std::string channelOutName = "kernel_output_ch";
    const std::string channelInName = "kernel_input_ch";

    virtual void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        bm->getExecutionSettings().programSettings->matrixSize = BLOCK_SIZE;
        data = bm->generateInputData();
        setupExternalChannelFiles();
    }

    virtual void TearDown() override {
        data = nullptr;
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

};


class LinpackKernelCommunicationTestLU : public LinpackKernelCommunicationTest {

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        executeKernel();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel kernel(*bm->getExecutionSettings().program, "lu", &err);

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, 0);
        err = kernel.setArg(2, 0);
        err = kernel.setArg(3, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(LU_BLOCK_OUT));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();
    }
};

class LinpackKernelCommunicationTestTop : public LinpackKernelCommunicationTest {

public: 
    std::vector<HOST_DATA_TYPE> lu_buffer_content;

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        // Generate uniformy distributed data
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        auto gefa_data = bm->generateInputData();
        linpack::gefa_ref_nopvt(gefa_data->A, bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
        // Fill all input channels with the correct number of 1.0s
        std::string fname = channelInName + std::to_string(2);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = (ii / CHUNK) * CHUNK; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&gefa_data->A[jj * bm->getExecutionSettings().programSettings->matrixSize + ii]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE*BLOCK_SIZE);
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel kernel(*bm->getExecutionSettings().program, "top_update", &err);

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, lu_buffer);
        err = kernel.setArg(2, CL_TRUE);
        err = kernel.setArg(3, 0);
        err = kernel.setArg(4, 0);
        err = kernel.setArg(5, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(TOP_BLOCK | TOP_BLOCK_OUT));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();

        lu_buffer_content.reserve(BLOCK_SIZE * BLOCK_SIZE);
        compute_queue.enqueueReadBuffer(lu_buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, lu_buffer_content.data());
 
    }
};


class LinpackKernelCommunicationTestTopOut : public LinpackKernelCommunicationTest {

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        // Generate uniformy distributed data
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        // Fill all input channels with the correct number of 1.0s
        std::string fname = channelInName + std::to_string(2);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE*BLOCK_SIZE);
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel kernel(*bm->getExecutionSettings().program, "top_update", &err);

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, lu_buffer);
        err = kernel.setArg(2, CL_FALSE);
        err = kernel.setArg(3, 0);
        err = kernel.setArg(4, 0);
        err = kernel.setArg(5, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(TOP_BLOCK_OUT));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);

        auto lu_data = bm->generateInputData();
        linpack::gefa_ref_nopvt(lu_data->A,bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
        std::vector<HOST_DATA_TYPE> lu_buffer_data(BLOCK_SIZE * BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j=(i/CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                lu_buffer_data[i * BLOCK_SIZE + j - (i/CHUNK)*CHUNK] = lu_data->A[j*BLOCK_SIZE + i];
            }
        }
        compute_queue.enqueueWriteBuffer(lu_buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, lu_buffer_data.data());
        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();
    }
};

class LinpackKernelCommunicationTestLeftOut : public LinpackKernelCommunicationTest {

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        // Generate uniformy distributed data
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        // Fill all input channels with the correct number of 1.0s
        std::string fname = channelInName + std::to_string(0);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE*BLOCK_SIZE);
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel kernel(*bm->getExecutionSettings().program, "left_update", &err);

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, lu_buffer);
        err = kernel.setArg(2, CL_FALSE);
        err = kernel.setArg(3, 0);
        err = kernel.setArg(4, 0);
        err = kernel.setArg(5, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(LEFT_BLOCK_OUT));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);

        auto lu_data = bm->generateInputData();
        linpack::gefa_ref_nopvt(lu_data->A,bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
        std::vector<HOST_DATA_TYPE> lu_buffer_data(BLOCK_SIZE * BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j=(i/CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                lu_buffer_data[i * BLOCK_SIZE + j - (i/CHUNK)*CHUNK] = lu_data->A[i*BLOCK_SIZE + j];
            }
        }
        compute_queue.enqueueWriteBuffer(lu_buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, lu_buffer_data.data());

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();
    }
};

class LinpackKernelCommunicationTestLeft : public LinpackKernelCommunicationTest {

    public: 
        std::vector<HOST_DATA_TYPE> lu_buffer_content;

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        // Generate uniformy distributed data
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        auto gefa_data = bm->generateInputData();
        linpack::gefa_ref_nopvt(gefa_data->A, bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
        // Fill all input channels with the correct number of 1.0s
        std::string fname = channelInName + std::to_string(0);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = (ii / CHUNK) * CHUNK; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&gefa_data->A[ii * bm->getExecutionSettings().programSettings->matrixSize + jj]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE*BLOCK_SIZE);
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE);
        cl::Kernel kernel(*bm->getExecutionSettings().program, "left_update", &err);

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, lu_buffer);
        err = kernel.setArg(2, CL_TRUE);
        err = kernel.setArg(3, 0);
        err = kernel.setArg(4, 0);
        err = kernel.setArg(5, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(LEFT_BLOCK | LEFT_BLOCK_OUT));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();

        lu_buffer_content.resize(BLOCK_SIZE * BLOCK_SIZE);
        compute_queue.enqueueReadBuffer(lu_buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, lu_buffer_content.data());
    }
};

class LinpackKernelCommunicationTestInner : public LinpackKernelCommunicationTest {

public:
    std::vector<HOST_DATA_TYPE> left_data;
    std::vector<HOST_DATA_TYPE> top_data;

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        // Generate uniformy distributed data
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        setupInputChannels();
        executeKernel();
    }

    void setupInputChannels() {
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
        auto left_data = bm->generateInputData();
        auto top_data = bm->generateInputData();
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        // Fill top channel with top result
        std::string fname = channelInName + std::to_string(1);
        std::remove(fname.c_str());
        std::ofstream fs;
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = 0; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&top_data->A[ii * bm->getExecutionSettings().programSettings->matrixSize + jj]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
        // Fill left channel with left result
        fname = channelInName + std::to_string(3);
        std::remove(fname.c_str());
        fs.open(fname, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        for (int ii = 0; ii < BLOCK_SIZE; ii++ ) {
            for (int jj = 0; jj < BLOCK_SIZE; jj++ ) {
                fs.write(reinterpret_cast<const char*>(&left_data->A[jj * bm->getExecutionSettings().programSettings->matrixSize + ii]), sizeof(HOST_DATA_TYPE));
            }
        }
        fs.close();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::CommandQueue network_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Kernel kernel(*bm->getExecutionSettings().program, "inner_update", &err);
        cl::Buffer top_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*BLOCK_SIZE);
        cl::Buffer left_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*BLOCK_SIZE);  
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE);      

        err = kernel.setArg(0, buffer);
        err = kernel.setArg(1, left_buffer_inner);
        err = kernel.setArg(2, top_buffer_inner);
        err = kernel.setArg(3, 0);
        err = kernel.setArg(4, 0);
        err = kernel.setArg(5, 1);

        // Start network layer kernel
        cl::Kernel network(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network.setArg(0, network_buffer);
        err = network.setArg(1, static_cast<cl_uint>(INNER_BLOCK));
        err = network.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        network_queue.enqueueNDRangeKernel(network, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

        network_queue.finish();

        left_data.resize(bm->getExecutionSettings().programSettings->matrixSize * BLOCK_SIZE);
        compute_queue.enqueueReadBuffer(left_buffer_inner, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, left_data.data());
        top_data.resize(bm->getExecutionSettings().programSettings->matrixSize * BLOCK_SIZE);
        compute_queue.enqueueReadBuffer(top_buffer_inner, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*BLOCK_SIZE * BLOCK_SIZE, top_data.data());
    }
};

class LinpackKernelCommunicationTestAll : public LinpackKernelCommunicationTest {

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        bm->getExecutionSettings().programSettings->matrixSize = 2 * BLOCK_SIZE;
        data = bm->generateInputData();
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
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer lu_buffer_top(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer top_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Buffer left_buffer_inner(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);        
        cl::Buffer network_buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*BLOCK_SIZE); 
        cl::Kernel innerkernel(*bm->getExecutionSettings().program, "inner_update", &err);

        err = innerkernel.setArg(0, buffer);
        err = innerkernel.setArg(1, left_buffer_inner);
        err = innerkernel.setArg(2, top_buffer_inner);
        err = innerkernel.setArg(3, 1);
        err = innerkernel.setArg(4, 1);
        err = innerkernel.setArg(5, 2);

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
        err = network1.setArg(1, static_cast<cl_uint>(INNER_BLOCK | LEFT_BLOCK | TOP_BLOCK| LEFT_BLOCK_OUT | TOP_BLOCK_OUT | LU_BLOCK_OUT));
        err = network1.setArg(2, NETWORK_FWD_TOP | NETWORK_FWD_RIGHT| NETWORK_FWD_BOTTOM | NETWORK_FWD_LEFT);
        cl::Kernel network2(*bm->getExecutionSettings().program, "network_layer", &err);
        err = network2.setArg(0, network_buffer);
        err = network2.setArg(1, static_cast<cl_uint>(LU_BLOCK_OUT));
        err = network2.setArg(2, 0);
        network_queue.enqueueNDRangeKernel(network1, cl::NullRange, cl::NDRange(1),cl::NullRange);
        network_queue.enqueueNDRangeKernel(network2, cl::NullRange, cl::NDRange(1),cl::NullRange);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueNDRangeKernel(lu1kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        left_queue.enqueueNDRangeKernel(leftkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        top_queue.enqueueNDRangeKernel(topkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.enqueueNDRangeKernel(innerkernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        inner_queue.finish();
        left_queue.finish();
        top_queue.finish();
        compute_queue.enqueueNDRangeKernel(lu2kernel, cl::NullRange, cl::NDRange(1),cl::NullRange);
        compute_queue.finish();
        network_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);

    }
};

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;

    // generate uniformly distributed block as top block
    auto ref_data = bm->generateInputData();

    linpack::gefa_ref_nopvt(ref_data->A, matrix_size, matrix_size);

    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToRightCorrectAmountOfData) {
    auto data_right = getDataFromExternalChannel(3, true);
    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_right.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(2, true);

    EXPECT_EQ(data_left.size(), BLOCK_SIZE * BLOCK_SIZE);
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0, true);

    EXPECT_EQ(data_left.size(), BLOCK_SIZE * BLOCK_SIZE);
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);
    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToRightCorrect) {
    // data that was sent to next top kernels
    auto data_left = getDataFromExternalChannel(3, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every row of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every column of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[2 * BLOCK_SIZE * j + i] - data_left[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToBottomCorrect) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
    if (data_top.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[j + i * 2 * BLOCK_SIZE] - data_top[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToLeftCorrect) {
    // data that was sent to kernels to the right
    auto data_left = getDataFromExternalChannel(2, true);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[i + (j + BLOCK_SIZE) * 2 * BLOCK_SIZE] - data_left[i*BLOCK_SIZE + j]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestAll, AllBlockExternalChannelOutputToTopCorrect) {
    // data that was sent to kernels below
    auto data_top = getDataFromExternalChannel(0, true);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_top.size(), number_values);
    if (data_top.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[(j + BLOCK_SIZE) + i * 2 * BLOCK_SIZE] - data_top[i*BLOCK_SIZE + j]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

// Start Unit tests for inner kernel

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;

    // generate uniformly distributed block as top block
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    auto ref_data = bm->generateInputData();
    auto left_data = bm->generateInputData();
    auto top_data = bm->generateInputData();
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;

    // do MM with left and top and add result to inner block
    for (int k = 0; k < matrix_size; k++) {
        for (int j = 0; j < matrix_size; j++) {
            for (int i = 0; i < matrix_size; i++) {
                ref_data->A[j * matrix_size + i] += top_data->A[k * matrix_size + i] * left_data->A[j * matrix_size + k];
            }
        }
    }
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockGlobalMemLeftBufferContentSameAsLeftChannel) {
    // data that was sent from LU kernel
    auto data_left = getDataFromExternalChannel(3, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }

    HOST_DATA_TYPE total_error = 0.0;
    // for every row of a block
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        // for every column of a block
        for (int j = 0; j < BLOCK_SIZE; j++) {
            total_error += std::abs(left_data[i * BLOCK_SIZE + j] - data_left[i * BLOCK_SIZE + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockGlobalMemTopBufferContentSameAsTopChannel) {
    // data that was sent from LU kernel
    auto data_top = getDataFromExternalChannel(1, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }

    HOST_DATA_TYPE total_error = 0.0;
    // for every row of a block
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        // for every column of a block
        for (int j = 0; j < BLOCK_SIZE; j++) {
            total_error += std::abs(top_data[i * BLOCK_SIZE + j] - data_top[i * BLOCK_SIZE + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToRightCorrectAmountOfData) {
    auto data_right = getDataFromExternalChannel(3, true);

    EXPECT_EQ(data_right.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(2, true);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0, true);
    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    EXPECT_EQ(data_top.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToTopCorrect) {
    // data that was sent to next top kernels
    auto data_bottom = getDataFromExternalChannel(0, true);
    // data that was sent from top kernel
    auto data_top = getDataFromExternalChannel(1, false);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_bottom.size(), number_values);
    if (data_bottom.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data_bottom[i + j * BLOCK_SIZE] - data_top[j*BLOCK_SIZE + i]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestInner, InnerBlockExternalChannelOutputToLeftCorrect) {
    // data that was sent to next top kernels
    auto data_right = getDataFromExternalChannel(2, true);
    // data that was sent from top kernel
    auto data_left = getDataFromExternalChannel(3, false);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_right.size(), number_values);
    if (data_right.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data_right[i + j * BLOCK_SIZE] - data_left[j*BLOCK_SIZE + i]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}


// START Unit tests for Left

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;
    auto gefa_data = bm->generateInputData();

    // generate uniformly distributed block as top block
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    auto ref_data = bm->generateInputData();
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
    linpack::gefa_ref_nopvt(gefa_data->A, matrix_size,matrix_size);

    // For each diagnonal element
    for (int k = 0; k < matrix_size; k++) {
        // For each row below the current row
        for (int j = 0; j < matrix_size; j++) {
            // multiply current column to current row and add it up
            for (int i = k + 1; i < matrix_size; i++) {
                ref_data->A[j * matrix_size + i] += ref_data->A[j * matrix_size + k] * gefa_data->A[k * matrix_size + i];
            }
        }
    }
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockGlobalMemLUBufferContentSameAsLUBlock) {
    // data that was sent from LU kernel
    auto data_lu = getDataFromExternalChannel(0, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }

    HOST_DATA_TYPE total_error = 0.0;

    size_t offset = 0;
    // for every row of a block
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        // for every column of a block
        for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
            total_error += std::abs(lu_buffer_content[i * BLOCK_SIZE + (j - (i / CHUNK) * CHUNK)] - data_lu[offset + (j - (i / CHUNK) * CHUNK)]);
        }
        offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToRightCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(3, true);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(2, true);
    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0, true);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToBottomCorrect) {
    // data that was sent to next top kernels
    auto data_bottom = getDataFromExternalChannel(1, true);
    // data that was sent from LU kernel
    auto data_lu = getDataFromExternalChannel(0, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_bottom.size(), number_values);
    if (data_bottom.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every row of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every column of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data_lu[offset + (j - (i / CHUNK) * CHUNK)] - data_bottom[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestLeft, LeftBlockExternalChannelOutputToLeftCorrect) {
    // data that was sent to kernels to the right
    auto data_left = getDataFromExternalChannel(2, true);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[i + j * BLOCK_SIZE] - data_left[i*BLOCK_SIZE + j]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;
    auto gefa_data = bm->generateInputData();

    // generate uniformly distributed block as top block
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    auto ref_data = bm->generateInputData();
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
    linpack::gefa_ref_nopvt(gefa_data->A, matrix_size,matrix_size);

	// std::cout << "Host:" << std::endl;
	// for (int i = 0; i < BLOCK_SIZE; i++) {
	// 	for (int j=0; j < BLOCK_SIZE; j++) {
	// 		std::cout << ref_data->A[i * matrix_size + j] << ",";
	// 	}
	// 	std::cout<< std::endl;
	// }
	// std::cout << std::endl;

    // std::cout << "Kernel:" << std::endl;
	// for (int i = 0; i < BLOCK_SIZE; i++) {
	// 	for (int j=0; j < BLOCK_SIZE; j++) {
	// 		std::cout << (data->A[i * matrix_size + j] - ref_data->A[i * matrix_size + j]) << ",";
	// 	}
	// 	std::cout<< std::endl;
	// }
	// std::cout << std::endl;

    // For each diagnonal element
    for (int k = 0; k < matrix_size; k++) {
        // For each element below it scale the current row
        for (int i = 0; i < matrix_size; i++) {
            ref_data->A[k * matrix_size + i] *= gefa_data->A[k * matrix_size + k];
        }
        // For each row below the current row
        for (int j = k + 1; j < matrix_size; j++) {
            // multiply current column to current row and add it up
            for (int i = 0; i < matrix_size; i++) {
                ref_data->A[j * matrix_size + i] += ref_data->A[k * matrix_size + i] * gefa_data->A[j * matrix_size + k];
            }
        }
    }
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToRightCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(3, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockGlobalMemLUBufferContentSameAsLUBlock) {
    // data that was sent from LU kernel
    auto data_lu = getDataFromExternalChannel(2, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }

    HOST_DATA_TYPE total_error = 0.0;

    size_t offset = 0;
    // for every row of a block
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        // for every column of a block
        for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
            total_error += std::abs(lu_buffer_content[i * BLOCK_SIZE + (j - (i / CHUNK) * CHUNK)] - data_lu[offset + (j - (i / CHUNK) * CHUNK)]);
        }
        offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(2, true);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0, true);
    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    EXPECT_EQ(data_top.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToRightCorrect) {
    // data that was sent to next top kernels
    auto data_left = getDataFromExternalChannel(3, true);
    // data that was sent from LU kernel
    auto data_lu = getDataFromExternalChannel(2, false);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every row of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every column of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data_lu[offset + (j - (i / CHUNK) * CHUNK)] - data_left[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestTop, TopBlockExternalChannelOutputToTopCorrect) {
    // data that was sent to kernels below
    auto data_top = getDataFromExternalChannel(0, true);

    size_t number_values = BLOCK_SIZE * BLOCK_SIZE;
    EXPECT_EQ(data_top.size(), number_values);
    if (data_top.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = 0; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[j + i * BLOCK_SIZE] - data_top[i*BLOCK_SIZE + j]);
            }
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalResultisSameAsRef) {
    auto data2 = bm->generateInputData();
    linpack::gefa_ref_nopvt(data2->A, bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(data2->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}


TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalResultisCorrect) {
    linpack::gesl_ref_nopvt(data->A, data->b, bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
    EXPECT_TRUE(bm->validateOutputAndPrintError(*data));

}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToRightCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(3, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(2, true);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0, true);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToRightCorrect) {
    // data that was sent to top kernels
    auto data_left = getDataFromExternalChannel(3, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every row of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every column of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[i + j * BLOCK_SIZE] - data_left[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToBottomCorrect) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(1, true);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
    if (data_top.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[j + i * BLOCK_SIZE] - data_top[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

// START Unit tests for Left without LU input over channels

TEST_F(LinpackKernelCommunicationTestLeftOut, LeftBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;
    auto gefa_data = bm->generateInputData();

    // generate uniformly distributed block as top block
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    auto ref_data = bm->generateInputData();
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
    linpack::gefa_ref_nopvt(gefa_data->A, matrix_size,matrix_size);

    // For each diagnonal element
    for (int k = 0; k < matrix_size; k++) {
        // For each row below the current row
        for (int j = 0; j < matrix_size; j++) {
            // multiply current column to current row and add it up
            for (int i = k + 1; i < matrix_size; i++) {
                ref_data->A[j * matrix_size + i] += ref_data->A[j * matrix_size + k] * gefa_data->A[k * matrix_size + i];
            }
        }
    }
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}


TEST_F(LinpackKernelCommunicationTestTopOut, TopBlockExternalResultisCorrect) {
    uint matrix_size = bm->getExecutionSettings().programSettings->matrixSize;
    auto gefa_data = bm->generateInputData();

    // generate uniformly distributed block as top block
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    auto ref_data = bm->generateInputData();
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
    linpack::gefa_ref_nopvt(gefa_data->A, matrix_size,matrix_size);

    // For each diagnonal element
    for (int k = 0; k < matrix_size; k++) {
        // For each element below it scale the current row
        for (int i = 0; i < matrix_size; i++) {
            ref_data->A[k * matrix_size + i] *= gefa_data->A[k * matrix_size + k];
        }
        // For each row below the current row
        for (int j = k + 1; j < matrix_size; j++) {
            // multiply current column to current row and add it up
            for (int i = 0; i < matrix_size; i++) {
                ref_data->A[j * matrix_size + i] += ref_data->A[k * matrix_size + i] * gefa_data->A[j * matrix_size + k];
            }
        }
    }
    double total_error = 0.0;
    for (int i = 0; i < bm->getExecutionSettings().programSettings->matrixSize; i++) {
        for (int j = 0; j < bm->getExecutionSettings().programSettings->matrixSize; j++) {
            total_error += std::abs(ref_data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j] - data->A[i * bm->getExecutionSettings().programSettings->matrixSize + j]);
        }
    }
    EXPECT_FLOAT_EQ(total_error, 0.0);
}

