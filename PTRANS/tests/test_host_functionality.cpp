//
// Created by Marius Meyer on 04.12.19.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "CL/cl.hpp"
#include "../src/host/transpose_functionality.hpp"
#include "gmock/gmock-matchers.h"

/**
 * Check correctness of reference matrix transposition implementation for 2x2
 */
TEST(TransposeFunctionality, make2x2MatrixTranspose) {

    size_t used_size = 2;
    auto A_test = new HOST_DATA_TYPE[used_size * used_size];
    auto B_test = new HOST_DATA_TYPE[used_size * used_size];
    auto result = new HOST_DATA_TYPE[used_size * used_size];
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            A_test[i * used_size + j] = (HOST_DATA_TYPE) (i * used_size + j);
            B_test[i * used_size + j] = 0.0;
        }
    }
    transposeReference(A_test, B_test, result, used_size);
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            EXPECT_FLOAT_EQ(A_test[i * used_size + j], result[j * used_size + i]);
        }
    }
}

/**
 * Check correctness of reference matrix transposition implementation for 9x9
 */
TEST(TransposeFunctionality, make9x9MatrixTranspose) {

    size_t used_size = 9;
    auto A_test = new HOST_DATA_TYPE[used_size * used_size];
    auto B_test = new HOST_DATA_TYPE[used_size * used_size];
    auto result = new HOST_DATA_TYPE[used_size * used_size];
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            A_test[i * used_size + j] = (HOST_DATA_TYPE) (i * used_size + j);
            B_test[i * used_size + j] = 0.0;
        }
    }
    transposeReference(A_test, B_test, result, used_size);
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            EXPECT_FLOAT_EQ(A_test[i * used_size + j], result[j * used_size + i]);
        }
    }
}

/**
 * Check that B is not transposed
 */
TEST(TransposeFunctionality, BStaysTheSame) {

    size_t used_size = 10;
    auto A_test = new HOST_DATA_TYPE[used_size * used_size];
    auto B_test = new HOST_DATA_TYPE[used_size * used_size];
    auto result = new HOST_DATA_TYPE[used_size * used_size];
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            B_test[i * used_size + j] = (HOST_DATA_TYPE) (i * used_size + j);
            A_test[i * used_size + j] = 0.0;
        }
    }
    transposeReference(A_test, B_test, result, used_size);
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            EXPECT_FLOAT_EQ(B_test[i * used_size + j], result[i * used_size + j]);
        }
    }
}

/**
 * Check if addition is done for A and B
 */
TEST(TransposeFunctionality, BAndAAreAddedUp) {

    size_t used_size = 10;
    auto A_test = new HOST_DATA_TYPE[used_size * used_size];
    auto B_test = new HOST_DATA_TYPE[used_size * used_size];
    auto result = new HOST_DATA_TYPE[used_size * used_size];
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            B_test[i * used_size + j] = (HOST_DATA_TYPE) (i * used_size + j);
            A_test[i * used_size + j] = 1.0;
        }
    }
    transposeReference(A_test, B_test, result, used_size);
    for (int i = 0; i < used_size; i++) {
        for (int j = 0; j < used_size; j++) {
            EXPECT_FLOAT_EQ(B_test[i * used_size + j] + 1.0, result[i * used_size + j]);
        }
    }
}

/**
 * Check if the output has the correct structure
 */
TEST(ResultOutput, OutputsCorrectFormatHeader) {
    std::vector<double> transferTimings;
    std::vector<double> calculateTimings;
    transferTimings.push_back(1.0);
    calculateTimings.push_back(1.0);
    std::shared_ptr<bm_execution::ExecutionTimings> results(
            new bm_execution::ExecutionTimings{transferTimings, calculateTimings});

    fpga_setup::setupEnvironmentAndClocks();

    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    printResults(results, 10);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex("(\\s+)trans(\\s+)calc(\\s+)calc\\sFLOPS(\\s+)total\\sFLOPS\n.*"));
}

/**
 * Check if the output values have correct formatting
 */
TEST(ResultOutput, OutputsCorrectFormatValues) {
    std::vector<double> transferTimings;
    std::vector<double> calculateTimings;
    transferTimings.push_back(1.0);
    calculateTimings.push_back(1.0);
    std::shared_ptr<bm_execution::ExecutionTimings> results(
            new bm_execution::ExecutionTimings{transferTimings, calculateTimings});

    fpga_setup::setupEnvironmentAndClocks();

    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    printResults(results, 10);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex(".*\navg:\\s+1\\.00000e\\+00\\s+1\\.00000e\\+00.*\n.*\n"));
}

/**
 * Checks if the error is printed to stdout and the error is aggregated over the whole matrix.
 */
TEST(ErrorOutput, AggregatedErrorIsPrinted) {
    HOST_DATA_TYPE *results = new HOST_DATA_TYPE[4 * 4];
    for (int i = 0; i < 4 * 4; i++) {
        results[i] = 0.0;
    }

    // Redirect stout buffer to local buffer to make checks possible
    std::stringstream newStdOutBuffer;
    std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(newStdOutBuffer.rdbuf());

    printCalculationError(4, results);

    // Redirect stdout to old buffer
    std::cout.rdbuf(oldStdOutBuffer);

    EXPECT_THAT(newStdOutBuffer.str(),
                ::testing::MatchesRegex("Maximum error:\\s+1\\.00000e\\+00\n"));
}

/**
 * Checks if the error is returned as an integer by the error calculation function.
 */
TEST(ErrorOutput, AggregatedErrorIsReturned) {
    HOST_DATA_TYPE *results = new HOST_DATA_TYPE[4 * 4];
    for (int i = 0; i < 4 * 4; i++) {
        results[i] = 0.0;
    }

    int error = printCalculationError(4, results);

    EXPECT_EQ(error, 1);
}

