//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>
#include "transpose_benchmark.hpp"
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"


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
            std::string fname = channelOutName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc);
            fs.close();
            std::remove((channelInName + std::to_string(i)).c_str());
            symlink(fname.c_str(), (channelInName + std::to_string(i)).c_str());
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
 * Checks the size and values of the timing measurements that are retured by calculate.
 */
TEST_F(TransposeKernelTest, FPGATimingsMeasuredForEveryIteration) {
    bm->getExecutionSettings().programSettings->numRepetitions = 10;
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(result->calculationTimings.size(), 10);
    EXPECT_EQ(result->transferTimings.size(), 10);
    for (int t = 0; t < 10; t++) {
        EXPECT_GE(result->transferTimings[t], 0.0);
        EXPECT_GE(result->calculationTimings[t], 0.0);
    }
}

/**
 * Check if the generated input data is in the specified range
 */
TEST_F(TransposeKernelTest, GenerateInputDataInRangeSingleBlock) {
    bm->getExecutionSettings().programSettings->matrixSize = 5;
    bm->getExecutionSettings().programSettings->blockSize = 5;
    auto data = bm->generateInputData();
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_LT(data->A[i * 5 + j], 100);
            EXPECT_GT(data->A[i * 5 + j], -100);
            EXPECT_LT(data->B[i * 5 + j], 101);
            EXPECT_GT(data->B[i * 5 + j], -99);
        }
    }
}

/**
 * Check if the generated input data is in the specified range
 */
TEST_F(TransposeKernelTest, GenerateInputDataInRangeMultipleBlocks) {
    bm->getExecutionSettings().programSettings->matrixSize = 6;
    bm->getExecutionSettings().programSettings->blockSize = 2;
    auto data = bm->generateInputData();
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_LT(data->A[i * 5 + j], 100);
            EXPECT_GT(data->A[i * 5 + j], -100);
            EXPECT_LT(data->B[i * 5 + j], 101);
            EXPECT_GT(data->B[i * 5 + j], -99);
        }
    }
}

