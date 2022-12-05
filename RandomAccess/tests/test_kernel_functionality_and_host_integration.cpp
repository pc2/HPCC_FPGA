//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "random_access_benchmark.hpp"
#include "test_program_settings.h"
#include "nlohmann/json.hpp"

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
    bm->executeKernel( *data);
    EXPECT_EQ(bm->getTimingsMap().at("execution").size(), 1);
}

/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_F(RandomAccessKernelTest, FPGACorrectNumberOfMeasurements3Rep) {
    bm->getExecutionSettings().programSettings->numRepetitions = 3;
    bm->executeKernel(*data);
    EXPECT_EQ(bm->getTimingsMap().at("execution").size(), 3);
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_F(RandomAccessKernelTest, FPGAErrorBelow1Percent) {
    bm->executeKernel(*data);
    bool success = bm->validateOutputAndPrintError(*data);
    EXPECT_TRUE(success);
}

using json = nlohmann::json;

TEST_F(RandomAccessKernelTest, JsonDump) {
    bm->executeKernel(*data);
    bm->collectResults();
    bm->dumpConfigurationAndResults("fft.json");
    std::FILE *f = std::fopen("fft.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        json j = json::parse(f);
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("timings")) {
            EXPECT_TRUE(j["timings"].contains("execution"));
        }
        EXPECT_TRUE(j.contains("results"));
        if (j.contains("results")) {
            EXPECT_TRUE(j["results"].contains("guops"));
            EXPECT_TRUE(j["results"].contains("t_mean"));
            EXPECT_TRUE(j["results"].contains("t_min"));
        }
    }
}
