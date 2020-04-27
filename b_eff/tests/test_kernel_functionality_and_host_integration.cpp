//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "../src/host/execution.h"
#include "parameters.h"
#include "../src/host/setup/fpga_setup.hpp"
#include "unistd.h"
#include "mpi.h"
#include <fstream>


struct OpenCLKernelTest : testing::Test {
    std::string kernelFileName = "communication_bw520n_emulate.aocx";
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    unsigned repetitions = 10;
    unsigned numberOfChannels = 4;
    std::string channelOutName = "kernel_output_ch";
    std::string channelInName = "kernel_input_ch";

    void createChannelFilesAndSymbolicLinks() {
        for (int i=0; i < numberOfChannels; i++) {
            std::string fname = channelOutName + std::to_string(i);
            std::fstream fs;
            fs.open(fname, std::ios::out);
            fs.close();
            std::remove((channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
            symlink(fname.c_str(), (channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
        }
    }

    void setupFPGA() {
        createChannelFilesAndSymbolicLinks();
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        repetitions
                });
    }
};

/**
 * Parametrized test takes a tuple of 4 parameters:
 * - name of the emulation bitstream
 * - number of channels
 * - name of the external output channel descriptors
 * - name of the external input channel descriptors
 */
struct DifferentOpenCLKernelTest : OpenCLKernelTest, testing::WithParamInterface<std::tuple<std::string, unsigned, std::string, std::string>> {
    DifferentOpenCLKernelTest() {
        auto params = GetParam();
        kernelFileName = std::get<0>(params);
        numberOfChannels = std::get<1>(params);
        channelOutName = std::get<2>(params);
        channelInName = std::get<3>(params);
        setupFPGA();
    }

    ~DifferentOpenCLKernelTest() {
        MPI_Finalize();
    }
};


/**
 * Tests if calculate returns the correct execution results
 */
TEST_P(DifferentOpenCLKernelTest, CalculateReturnsCorrectExecutionResultFor111) {
    config->repetitions = 1;
    auto result = bm_execution::calculate(config, 1,1);
    EXPECT_EQ(1, result->messageSize);
    EXPECT_EQ(1, result->looplength);
    EXPECT_EQ(1, result->calculationTimings.size());
}

/**
 * Tests if calculate returns the correct execution results for multiple repetitions
 */
TEST_P(DifferentOpenCLKernelTest, CalculateReturnsCorrectExecutionResultFor842) {
    config->repetitions = 2;
    auto result = bm_execution::calculate(config, 8,4);
    EXPECT_EQ(8, result->messageSize);
    EXPECT_EQ(4, result->looplength);
    EXPECT_EQ(2, result->calculationTimings.size());
}

/**
 * Tests if data is written to the channels for small message sizes
 */
TEST_P(DifferentOpenCLKernelTest, DataIsWrittenToChannelForMessageSizeFillingOneChannel) {
    config->repetitions = 1;
    const unsigned messageSize = CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    auto result = bm_execution::calculate(config, messageSize,looplength);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[messageSize * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), messageSize * looplength * 2);
        fs.close();
        // Altough only one channel would be necessary to send the data, also a dummy is send over the second channel
        // to simplify the kernel logic
        EXPECT_EQ( messageSize * looplength ,num_bytes);

    }
    delete [] buffer;
}

/**
 * Tests if data is written to the channels for small message sizes filling two channels
 */
TEST_P(DifferentOpenCLKernelTest, DataIsWrittenToChannelForMessageSizeFillingTwoChannels) {
    config->repetitions = 1;
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    auto result = bm_execution::calculate(config, messageSize,looplength);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[messageSize * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), messageSize * looplength * 2);
        fs.close();
        EXPECT_FLOAT_EQ( messageSize * looplength / 2 ,num_bytes);
    }
    delete [] buffer;
}

/**
 * Tests if data is written to the channels for message sizes filling more than two channels
 */
TEST_P(DifferentOpenCLKernelTest, DataIsWrittenToChannelForMessageSizeFillingMoreThanTwoChannels) {
    config->repetitions = 1;
    const unsigned messageSize = 4 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 1;
    auto result = bm_execution::calculate(config, messageSize,looplength);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[messageSize * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), messageSize * looplength * 2);
        fs.close();
        EXPECT_FLOAT_EQ( messageSize * looplength / 2,num_bytes);
    }
    delete [] buffer;
}

/**
 * Tests if correct data is written to the channels
 */
TEST_P(DifferentOpenCLKernelTest, CorrectDataIsWrittenToChannel) {
    config->repetitions = 1;
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    auto result = bm_execution::calculate(config, messageSize,looplength);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[messageSize * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), messageSize * looplength * 2);
        fs.close();
        for (int k=0; k < messageSize * looplength / 2; k++) {
            EXPECT_EQ(static_cast<HOST_DATA_TYPE>(messageSize & 255), buffer[k]);
        }
    }
    delete [] buffer;
}



INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
        testing::Values(std::make_tuple("communication_bw520n_emulate.aocx", 4, "kernel_output_ch", "kernel_input_ch")));
