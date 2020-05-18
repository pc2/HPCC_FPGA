//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "random_access_benchmark.hpp"
#include "test_program_settings.h"


struct RandomAccessKernelTest : testing::Test {
    std::shared_ptr<random_access::RandomAccessData> data;

    RandomAccessKernelTest() {
        bm->getExecutionSettings().programSettings->dataSize =  128 * NUM_KERNEL_REPLICATIONS * BUFFER_SIZE;
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
    }

    void SetUp() override {
        data = bm->generateInputData(bm->getExecutionSettings());
    }

};


/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_F(RandomAccessKernelTest, FPGACorrectNumberOfMeasurements1Rep) {
    auto result = bm->executeKernel(bm->getExecutionSettings(), *data);
    EXPECT_EQ(result->times.size(), 1);
}

/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_F(RandomAccessKernelTest, FPGACorrectNumberOfMeasurements3Rep) {
    bm->getExecutionSettings().programSettings->numRepetitions = 3;
    auto result = bm->executeKernel(bm->getExecutionSettings(), *data);
    EXPECT_EQ(result->times.size(), 3);
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_F(RandomAccessKernelTest, FPGAErrorBelow1Percent) {
    auto result = bm->executeKernel(bm->getExecutionSettings(), *data);
    bool success = bm->validateOutputAndPrintError(bm->getExecutionSettings(), *data, *result);
    EXPECT_TRUE(success);
}
