//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "random_access_benchmark.hpp"
#include "test_program_settings.h"


struct RandomAccessKernelTest : testing::Test {
    std::unique_ptr<random_access::RandomAccessData> data;
    std::unique_ptr<random_access::RandomAccessBenchmark> bm;

    RandomAccessKernelTest() {
        bm = std::unique_ptr<random_access::RandomAccessBenchmark>(new random_access::RandomAccessBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->dataSize =  128 * NUM_REPLICATIONS * BUFFER_SIZE;
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
    }

    void SetUp() override {
        data = bm->generateInputData();
    }

};


/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_F(RandomAccessKernelTest, FPGACorrectNumberOfMeasurements1Rep) {
    auto result = bm->executeKernel( *data);
    EXPECT_EQ(result->times.size(), 1);
}

/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_F(RandomAccessKernelTest, FPGACorrectNumberOfMeasurements3Rep) {
    bm->getExecutionSettings().programSettings->numRepetitions = 3;
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(result->times.size(), 3);
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_F(RandomAccessKernelTest, FPGAErrorBelow1Percent) {
    auto result = bm->executeKernel(*data);
    bool success = bm->validateOutputAndPrintError(*data);
    EXPECT_TRUE(success);
}
