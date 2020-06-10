//
// Created by Marius Meyer on 04.12.19.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "CL/cl.hpp"
#include "test_program_settings.h"
#include "gmock/gmock-matchers.h"
#include "transpose_benchmark.hpp"


struct TransposeHostTest : testing::Test {
    std::unique_ptr<transpose::TransposeBenchmark> bm;

    TransposeHostTest() {
        bm = std::unique_ptr<transpose::TransposeBenchmark>( new transpose::TransposeBenchmark(global_argc, global_argv));
    }
};

/**
 * Check if the output has the correct structure
 */
TEST_F(TransposeHostTest, OutputsCorrectFormatHeader) {
    std::vector<double> transferTimings;
    std::vector<double> calculateTimings;
    transferTimings.push_back(1.0);
    calculateTimings.push_back(1.0);
    std::shared_ptr<transpose::TransposeExecutionTimings> results(
            new transpose::TransposeExecutionTimings{transferTimings, calculateTimings});


    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    bm->printResults(*results);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex("(\\s+)trans(\\s+)calc(\\s+)calc\\sFLOPS(\\s+)total\\sFLOPS\n.*"));
}

/**
 * Check if the output values have correct formatting
 */
TEST_F(TransposeHostTest, OutputsCorrectFormatValues) {
    std::vector<double> transferTimings;
    std::vector<double> calculateTimings;
    transferTimings.push_back(1.0);
    calculateTimings.push_back(1.0);
    std::shared_ptr<transpose::TransposeExecutionTimings> results(
            new transpose::TransposeExecutionTimings{transferTimings, calculateTimings});


    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    bm->printResults(*results);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex(".*\navg:\\s+1\\.00000e\\+00\\s+1\\.00000e\\+00.*\n.*\n"));
}

/**
 * Checks if the error is printed to stdout and the error is aggregated over the whole matrix.
 */
TEST_F(TransposeHostTest, AggregatedErrorIsPrinted) {
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    bm->executeBenchmark();
    auto data = bm->generateInputData();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            data->A[i * 4 + j] = i * 4 + j;
            data->B[i * 4 + j] = i * 4 + j;
        }
    }

    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    bool success = bm->validateOutputAndPrintError(*data);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex("Maximum error:\\s+3\\.00000e\\+01\n"));
    EXPECT_FALSE(success);
}

/**
 * Checks if the error is printed to stdout and validation can be success.
 */
TEST_F(TransposeHostTest, ValidationIsSuccess) {
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    bm->executeBenchmark();
    auto data = bm->generateInputData();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            data->A[i * 4 + j] = 0.0;
            data->B[i * 4 + j] = 0.0;
        }
    }

    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    bool success = bm->validateOutputAndPrintError(*data);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex("Maximum error:\\s+0\\.00000e\\+00\n"));
    EXPECT_TRUE(success);
}


