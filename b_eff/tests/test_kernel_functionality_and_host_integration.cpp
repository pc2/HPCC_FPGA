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

    NetworkKernelTest() {}

    void SetUp() override {
        bm = std::unique_ptr<network::NetworkBenchmark>(new network::NetworkBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
        data = bm->generateInputData();
        createChannelFilesAndSymbolicLinks();
    }

    void TearDown() override {
        bm = nullptr;
        data = nullptr;
    }

    void createChannelFilesAndSymbolicLinks() {
        for (int i=0; i < numberOfChannels; i++) {
            std::string fname = channelOutName + std::to_string(i);
            std::remove(fname.c_str());
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
    if (bm->getExecutionSettings().programSettings->communicationType == hpcc_base::CommunicationType::intel_external_channels) {
        // Skip this test if no IEC are used, because they are specific to the IEC emulation based on files
        GTEST_SKIP() << "Intel external channel needs at least message size of 64 byte to fill channel!";
    }
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(1,1, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    EXPECT_NE(bm->collected_timings.end(), bm->collected_timings.find(1));
    EXPECT_EQ(1, bm->collected_timings.find(1)->second.execution_timings.at(0).looplength);
    EXPECT_EQ(1, bm->collected_timings.find(1)->second.execution_timings.at(0).calculationTimings.size());
}

/**
 * Tests if calculate returns the correct execution results for multiple repetitions
 */
TEST_F(NetworkKernelTest, CalculateReturnsCorrectExecutionResultFor842) {
    bm->getExecutionSettings().programSettings->numRepetitions = 2;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(8,4, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    EXPECT_NE(bm->collected_timings.end(), bm->collected_timings.find(8));
    EXPECT_EQ(4, bm->collected_timings.find(8)->second.execution_timings.at(0).looplength);
    EXPECT_EQ(2, bm->collected_timings.find(8)->second.execution_timings.at(0).calculationTimings.size());
}

/**
 * Tests if data is written to the channels for small message sizes
 */
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingOneChannel) {
    if (bm->getExecutionSettings().programSettings->communicationType != hpcc_base::CommunicationType::intel_external_channels) {
        // Skip this test if no IEC are used, because they are specific to the IEC emulation based on files
        GTEST_SKIP();
    }
    const unsigned messageSize = std::log2(CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[(1 << messageSize) * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), (1 << messageSize) * looplength * 2);
        fs.close();
        // Altough only one channel would be necessary to send the data, also a dummy is send over the second channel
        // to simplify the kernel logic
        EXPECT_EQ( (1 << messageSize) * looplength ,num_bytes);

    }
    delete [] buffer;
}

/**
 * Tests if data is written to the channels for small message sizes filling two channels
 */
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingTwoChannels) {
    if (bm->getExecutionSettings().programSettings->communicationType != hpcc_base::CommunicationType::intel_external_channels) {
        // Skip this test if no IEC are used, because they are specific to the IEC emulation based on files
        GTEST_SKIP();
    }
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize, looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[(1 << messageSize) * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), (1 << messageSize) * looplength * 2);
        fs.close();
        EXPECT_FLOAT_EQ((1 << messageSize) * looplength / 2 ,num_bytes);
    }
    delete [] buffer;
}

/**
 * Tests if data is written to the channels for message sizes filling more than two channels
 */
TEST_F(NetworkKernelTest, DataIsWrittenToChannelForMessageSizeFillingMoreThanTwoChannels) {
    if (bm->getExecutionSettings().programSettings->communicationType != hpcc_base::CommunicationType::intel_external_channels) {
        // Skip this test if no IEC are used, because they are specific to the IEC emulation based on files
        GTEST_SKIP();
    }
    const unsigned messageSize = std::log2(8 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    HOST_DATA_TYPE* buffer = new HOST_DATA_TYPE[(1 << messageSize) * looplength * 2];
    for (int i=0; i < numberOfChannels; i++) {
        std::string ifname = channelOutName + std::to_string(i);
        std::fstream fs;
        fs.open(ifname, std::ios::in | std::ios::binary);
        int num_bytes = fs.readsome(reinterpret_cast<char*>(buffer), (1 << messageSize) * looplength * 2);
        fs.close();
        EXPECT_FLOAT_EQ((1 << messageSize) * looplength / 2,num_bytes);
    }
    delete [] buffer;
}

/**
 * Tests if correct data is written to the channels
 */
TEST_F(NetworkKernelTest, CorrectDataIsWrittenToChannel) {
    if (bm->getExecutionSettings().programSettings->communicationType != hpcc_base::CommunicationType::intel_external_channels) {
        // Skip this test if no IEC are used, because they are specific to the IEC emulation based on files
        GTEST_SKIP();
    }
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
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
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    HOST_DATA_TYPE cvalue = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    EXPECT_EQ(cvalue, data->items[0].validationBuffer[0]);
    bool all_same = true;
    for (int i = 0; i < data->items[0].validationBuffer.size(); i++) {
        all_same = all_same & (data->items[0].validationBuffer[0] == data->items[0].validationBuffer[i]);
    }
    EXPECT_TRUE(all_same);
}

TEST_F(NetworkKernelTest, ValidationDataIsStoredCorrectlyForSmallMessageSize) {
    if (bm->getExecutionSettings().programSettings->communicationType == hpcc_base::CommunicationType::intel_external_channels) {
        GTEST_SKIP() << "Intel external channel needs at least message size of 64 byte to fill channel!";
    }
    const unsigned messageSize = 0;
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    HOST_DATA_TYPE cvalue = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    EXPECT_EQ(cvalue, data->items[0].validationBuffer[0]);
    bool all_same = true;
    for (int i = 0; i < data->items[0].validationBuffer.size(); i++) {
        all_same = all_same & (data->items[0].validationBuffer[0] == data->items[0].validationBuffer[i]);
    }
    EXPECT_TRUE(all_same);
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForLoopLength4) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    const unsigned looplength = 4;
    const unsigned replications = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, 1));
    EXPECT_EQ((1 << messageSize), data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForLoopLength1) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, 1));
    EXPECT_EQ((1 << messageSize), data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForDifferentMessageSize) {
    const unsigned messageSize = 0;
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    const unsigned looplength = 1;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    EXPECT_EQ((1 << messageSize), data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataHasCorrectSizeForReplication2) {
    const unsigned messageSize = 4;
    const unsigned looplength = 2;
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, 2));
    EXPECT_EQ((1 << messageSize) * 2, data->items[0].validationBuffer.size());
}

TEST_F(NetworkKernelTest, ValidationDataSingleItemWrongCheckFails) {
    const unsigned messageSize = 4;
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data;});
    data->items[0].validationBuffer[looplength] = expected_data + 1;
    EXPECT_FALSE(bm->validateOutput(*data));
    bm->printError();
}

TEST_F(NetworkKernelTest, ValidationDataWrongCheckFails) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data - 1;});
    EXPECT_FALSE(bm->validateOutput(*data));
    bm->printError();
}

TEST_F(NetworkKernelTest, ValidationDataCorrectCheckSuccessful) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const HOST_DATA_TYPE expected_data = static_cast<HOST_DATA_TYPE>(messageSize & 255);
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    std::for_each(data->items[0].validationBuffer.begin(), data->items[0].validationBuffer.end(), [expected_data](HOST_DATA_TYPE& d){d = expected_data;});
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError();
}

TEST_F(NetworkKernelTest, ValidationDataCorrectOneMessageSizeAfterExecution) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError();
}

// This test is disabled because it does not work with the current implementation of the
// external channels in software emulation. The different kernel executions will read 
// the old data from the channel file, which will lead to a failing validation!
TEST_F(NetworkKernelTest, ValidationDataCorrectTwoMessageSizesAfterExecution) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize + 1,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError();
}

TEST_F(NetworkKernelTest, ValidationDataWrongTwoMessageSizesAfterExecution) {
    const unsigned messageSize = std::log2(2 * CHANNEL_WIDTH / sizeof(HOST_DATA_TYPE));
    const unsigned looplength = 4;
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    data->items.push_back(network::NetworkData::NetworkDataItem(messageSize + 1,looplength, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    data->items[1].validationBuffer[0] = static_cast<HOST_DATA_TYPE>(0);
    EXPECT_FALSE(bm->validateOutput(*data));
    bm->printError();
}

TEST_F(NetworkKernelTest, JsonDump) {
    data->items.clear();
    data->items.push_back(network::NetworkData::NetworkDataItem(8,4, bm->getExecutionSettings().programSettings->kernelReplications));
    bm->executeKernel(*data);
    bm->collectResults();
    bm->dumpConfigurationAndResults("b_eff.json");
    std::FILE *f = std::fopen("b_eff.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        json j = json::parse(f);
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("timings")) {
            EXPECT_TRUE(j["timings"].size() > 0);
            if (j["timings"].size() > 0) {
                for (const auto& timing: j["timings"].items()) {
                    EXPECT_TRUE(timing.value().contains("maxCalcBW"));
                    EXPECT_TRUE(timing.value().contains("maxMinCalculationTime"));
                    EXPECT_TRUE(timing.value().contains("timings"));
                    if (timing.value().contains("timings")) {
                        for (const auto& timing: timing.value()["timings"]) {
                            EXPECT_TRUE(timing.contains("looplength"));
                            EXPECT_TRUE(timing.contains("messageSize"));
                            EXPECT_TRUE(timing.contains("timings"));
                        }
                    }
                }
            }
        }
        EXPECT_TRUE(j.contains("results"));
        if (j.contains("results")) {
            EXPECT_TRUE(j["results"].contains("b_eff"));
        }
    }
}
