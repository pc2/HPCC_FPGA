//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>
#include "transpose_benchmark.hpp"
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "nlohmann/json.hpp"

struct TransposeKernelTest : testing::Test {
    std::shared_ptr<transpose::TransposeData> data;
    std::unique_ptr<transpose::TransposeBenchmark> bm;
    uint matrix_size = BLOCK_SIZE;
    unsigned numberOfChannels = 4;
    std::string channelOutName = "kernel_output_ch";
    std::string channelInName = "kernel_input_ch";

    TransposeKernelTest() {
        bm = std::unique_ptr<transpose::TransposeBenchmark>( new transpose::TransposeBenchmark(global_argc, global_argv));
    }

    void SetUp() override {
        matrix_size = BLOCK_SIZE;
        bm->getExecutionSettings().programSettings->matrixSize = matrix_size;
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
        bm->getExecutionSettings().programSettings->kernelReplications = 1;
        data = bm->generateInputData();
        createChannelFilesAndSymbolicLinks();
    }

    void createChannelFilesAndSymbolicLinks() {
        for (int i=0; i < numberOfChannels; i++) {
            // Combine 0+1,2+3,...
            int partnerChannelId = 2 * (i / 2) + ((i + 1) % 2); 
            std::string fname = channelOutName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc);
            fs.close();
            std::remove((channelInName + std::to_string(partnerChannelId)).c_str());
            symlink(fname.c_str(), (channelInName + std::to_string(partnerChannelId)).c_str());
        }
    }
};


/**
 * Tests if B will not be transposed
 */
TEST_F(TransposeKernelTest, FPGACorrectBStaysTheSame) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = 0.0;
            data->B[i * matrix_size + j] = i * matrix_size + j;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - data->B[i * matrix_size + j]);
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

/**
 * Tests if B will not be transposed
 */
TEST_F(TransposeKernelTest, FPGACorrectBStaysTheSame4Blocks) {
    if (bm->getExecutionSettings().programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
        // Diagonal data handler does not support this test, since matrix is stored differently in memory buffer
        return;
    }
    matrix_size = BLOCK_SIZE * bm->getExecutionSettings().programSettings->kernelReplications;
    bm->getExecutionSettings().programSettings->matrixSize = matrix_size;
    data = bm->generateInputData();
    createChannelFilesAndSymbolicLinks();
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = 0.0;
            data->B[i * matrix_size + j] = i * matrix_size + j;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - data->B[i * matrix_size + j]);
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

/**
 * Tests if a block of A will be correctly transposed
 */
TEST_F(TransposeKernelTest, FPGAABlockIsTransposed) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = i * matrix_size + j;
            data->B[i * matrix_size + j] = 0.0;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - data->A[j * matrix_size + i]);
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

/**
 * Tests if a block of A will be correctly transposed
 */
TEST_F(TransposeKernelTest, FPGAABlockIsTransposed4Blocks) {
    if (bm->getExecutionSettings().programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
        // Diagonal data handler does not support this test, since matrix is stored differently in memory buffer
        return;
    }
    matrix_size = BLOCK_SIZE * bm->getExecutionSettings().programSettings->kernelReplications;
    bm->getExecutionSettings().programSettings->matrixSize = matrix_size;
    data = bm->generateInputData();
    createChannelFilesAndSymbolicLinks();
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = i * matrix_size + j;
            data->B[i * matrix_size + j] = 0.0;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - data->A[j * matrix_size + i]);
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

/**
 * Tests if matrix A and B will be summed up in the result
 */
TEST_F(TransposeKernelTest, FPGAAAndBAreSummedUp) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = 1.0;
            data->B[i * matrix_size + j] = i * matrix_size + j;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - (data->B[i * matrix_size + j] + 1.0));
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

/**
 * Tests if matrix A and B will be summed up in the result
 */
TEST_F(TransposeKernelTest, FPGAAAndBAreSummedUp4Blocks) {
    if (bm->getExecutionSettings().programSettings->dataHandlerIdentifier == transpose::data_handler::DataHandlerType::diagonal) {
        // Diagonal data handler does not support this test, since matrix is stored differently in memory buffer
        return;
    }
    matrix_size = BLOCK_SIZE * bm->getExecutionSettings().programSettings->kernelReplications;
    bm->getExecutionSettings().programSettings->matrixSize = matrix_size;
    data = bm->generateInputData();
    createChannelFilesAndSymbolicLinks();
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = 1.0;
            data->B[i * matrix_size + j] = i * matrix_size + j;
        }
    }
    bm->executeKernel(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            aggregated_error += std::abs(data->result[i * matrix_size + j] - (data->B[i * matrix_size + j] + 1.0));
        }
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}


/**
 * Checks the size and values of the timing measurements that are retured by calculate.
 */
TEST_F(TransposeKernelTest, FPGATimingsMeasuredForEveryIteration) {
    bm->getExecutionSettings().programSettings->numRepetitions = 10;
    bm->executeKernel(*data);
    EXPECT_EQ(bm->getTimingsMap().at("calculation").size(), 10);
    EXPECT_EQ(bm->getTimingsMap().at("transfer").size(), 10);
    for (int t = 0; t < 10; t++) {
        EXPECT_GE(bm->getTimingsMap().at("transfer")[t], 0.0);
        EXPECT_GE(bm->getTimingsMap().at("calculation")[t], 0.0);
    }
}

using json = nlohmann::json;

TEST_F(TransposeKernelTest, JsonDump) {
    bm->executeKernel(*data);
    bm->collectResults();
    bm->dumpConfigurationAndResults("ptrans.json");
    std::FILE *f = std::fopen("ptrans.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        json j = json::parse(f);
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("timings")) {
            EXPECT_TRUE(j["timings"].contains("calculation"));
            EXPECT_TRUE(j["timings"].contains("transfer"));
        }
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("results")) {
            EXPECT_TRUE(j["results"].contains("avg_calc_flops"));
            EXPECT_TRUE(j["results"].contains("avg_calc_t"));
            EXPECT_TRUE(j["results"].contains("avg_mem_bandwidth"));
            EXPECT_TRUE(j["results"].contains("avg_t"));
            EXPECT_TRUE(j["results"].contains("avg_transfer_bandwidth"));
            EXPECT_TRUE(j["results"].contains("avg_transfer_t"));
            EXPECT_TRUE(j["results"].contains("max_calc_flops"));
            EXPECT_TRUE(j["results"].contains("max_mem_bandwidth"));
            EXPECT_TRUE(j["results"].contains("max_transfer_bandwidth"));
            EXPECT_TRUE(j["results"].contains("min_calc_t"));
            EXPECT_TRUE(j["results"].contains("min_t"));
            EXPECT_TRUE(j["results"].contains("min_transfer_t"));
        }
    }
}
