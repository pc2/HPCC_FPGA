//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "stream_benchmark.hpp"


struct StreamKernelTest :public  ::testing::Test {
    std::shared_ptr<stream::StreamData> data;
    std::unique_ptr<stream::StreamBenchmark> bm;

    StreamKernelTest() {
        bm = std::unique_ptr<stream::StreamBenchmark>(new stream::StreamBenchmark(global_argc, global_argv));
    }

    void SetUp( ) { 
        bm->getExecutionSettings().programSettings->streamArraySize = VECTOR_COUNT * UNROLL_COUNT * NUM_REPLICATIONS * BUFFER_SIZE;
        data = bm->generateInputData();
   }


};


/**
 * Execution returns correct results for a single repetition
 */
TEST_F(StreamKernelTest, FPGACorrectResultsOneRepetition) {
    bm->getExecutionSettings().programSettings->numRepetitions = 1;
    auto result = bm->executeKernel(*data);
    for (int i = 0; i < bm->getExecutionSettings().programSettings->streamArraySize; i++) {
        EXPECT_FLOAT_EQ(data->A[i], 30.0);
        EXPECT_FLOAT_EQ(data->B[i], 6.0);
        EXPECT_FLOAT_EQ(data->C[i], 8.0);
    }
}

/**
 * Execution returns correct results for three repetitions
 */
TEST_F(StreamKernelTest, FPGACorrectResultsThreeRepetition) {
    bm->getExecutionSettings().programSettings->numRepetitions = 3;
    auto result = bm->executeKernel(*data);
    for (int i = 0; i < bm->getExecutionSettings().programSettings->streamArraySize; i++) {
        EXPECT_FLOAT_EQ(data->A[i], 6750.0);
        EXPECT_FLOAT_EQ(data->B[i], 1350.0);
        EXPECT_FLOAT_EQ(data->C[i], 1800.0);
    }
}
