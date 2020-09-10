//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "network_benchmark.hpp"
#include "parameters.h"
#include "mpi.h"
#include "test_program_settings.h"
#include <fstream>

struct NetworkKernelTest : testing::Test {
    std::unique_ptr<network::NetworkBenchmark> bm;
    std::unique_ptr<network::NetworkData> data;
    unsigned numberOfChannels = 4;
    std::string channelOutName = "kernel_output_ch";
    std::string channelInName = "kernel_input_ch";

    NetworkKernelTest() {
        bm = std::unique_ptr<network::NetworkBenchmark>(new network::NetworkBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
        data = bm->generateInputData();
    }

    void SetUp() override {
        createChannelFilesAndSymbolicLinks();
    }

    void createChannelFilesAndSymbolicLinks() {
        for (int i=0; i < numberOfChannels; i++) {
            std::string fname = channelOutName + std::to_string(i);
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc);
            fs.close();
            std::remove((channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
            symlink(fname.c_str(), (channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
        }
    }
};

/**
 * Tests if calculate returns the correct execution results
 */
TEST_F(NetworkKernelTest, CalculateReturnsCorrectExecutionResultFor111) {
    bm->getExecutionSettings().programSettings->numRepetitions = 1;
    bm->getExecutionSettings().programSettings->looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(1,1));
    auto result = bm->executeKernel(*data);
    EXPECT_NE(result->timings.end(), result->timings.find(1));
    EXPECT_EQ(1, result->timings.find(1)->second->at(0)->looplength);
    EXPECT_EQ(1, result->timings.find(1)->second->at(0)->calculationTimings.size());
}

/**
 * Tests if calculate returns the correct execution results for multiple repetitions
 */
TEST_F(NetworkKernelTest, CalculateReturnsCorrectExecutionResultFor842) {
    bm->getExecutionSettings().programSettings->numRepetitions = 2;
    bm->getExecutionSettings().programSettings->looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(8,4));
    auto result = bm->executeKernel(*data);
    EXPECT_NE(result->timings.end(), result->timings.find(8));
    EXPECT_EQ(4, result->timings.find(8)->second->at(0)->looplength);
    EXPECT_EQ(2, result->timings.find(8)->second->at(0)->calculationTimings.size());
}

/**
 * Tests if data is written to the channels for small message sizes
 */
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingOneChannel) {
    const unsigned messageSize = CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
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
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingTwoChannels) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize, looplength));
    auto result = bm->executeKernel(*data);
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
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingMoreThanTwoChannels) {
    const unsigned messageSize = 4 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
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
TEST_F(NetworkKernelTest, CorrectDataIsWrittenToChannel) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
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

TEST_F(NetworkKernelTest, ValidationDataIsStoredCorrectlyForTwoChannels) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
    HOST_DATA_TYPE cvalue = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    EXPECT_EQ(cvalue, data->items[0].validationBuffer[0]);
    bool all_same = true;
    for (int i = 0; i < data->items[0].validationBuffer.size(); i++) {
        all_same = all_same & (data->items[0].validationBuffer[0] == data->items[0].validationBuffer[i]);
    }
    EXPECT_TRUE(all_same);
}

TEST_F(NetworkKernelTest, ValidationDataIsStoredCorrectlyForSmallMessageSize) {
    const unsigned messageSize = 1;
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
    HOST_DATA_TYPE cvalue = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    EXPECT_EQ(cvalue, data->items[0].validationBuffer[0]);
    bool all_same = true;
    for (int i = 0; i < data->items[0].validationBuffer.size(); i++) {
        all_same = all_same & (data->items[0].validationBuffer[0] == data->items[0].validationBuffer[i]);
    }
    EXPECT_TRUE(all_same);
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForLoopLength4) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(looplength * CHANNEL_WIDTH * 2 * 2, data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForLoopLength1) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(looplength * CHANNEL_WIDTH * 2 * 2, data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForDiffernetMessageSize) {
    const unsigned messageSize = 1;
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(looplength * CHANNEL_WIDTH * 2 * 2, data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataSingleItemWrongCheckFails) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data;});
    data->items[0].validationBuffer[looplength] = expected_data + 1;
    EXPECT_FALSE(bm->validateOutputAndPrintError(*data));
}

TEST_F(NetworkKernelTest, ValidationDataWrongCheckFails) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data - 1;});
    EXPECT_FALSE(bm->validateOutputAndPrintError(*data));
}

TEST_F(NetworkKernelTest, ValidationDataCorrectCheckSuccessful) {
    const unsigned messageSize = 2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE);
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data;});
    EXPECT_TRUE(bm->validateOutputAndPrintError(*data));
}

