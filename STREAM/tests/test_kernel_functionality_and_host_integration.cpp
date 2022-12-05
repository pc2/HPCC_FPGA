//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "stream_benchmark.hpp"
#include "nlohmann/json.hpp"

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
    bm->executeKernel(*data);
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
    bm->executeKernel(*data);
    for (int i = 0; i < bm->getExecutionSettings().programSettings->streamArraySize; i++) {
        EXPECT_FLOAT_EQ(data->A[i], 6750.0);
        EXPECT_FLOAT_EQ(data->B[i], 1350.0);
        EXPECT_FLOAT_EQ(data->C[i], 1800.0);
    }
}

using json = nlohmann::json;

TEST_F(StreamKernelTest, JsonDump) {
    bm->executeKernel(*data);
    bm->collectResults();
    bm->dumpConfigurationAndResults("stream.json");
    std::FILE *f = std::fopen("stream.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        json j = json::parse(f);
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("timings")) {
            EXPECT_TRUE(j["timings"].contains("Add"));
            EXPECT_TRUE(j["timings"].contains("Copy"));
            EXPECT_TRUE(j["timings"].contains("PCI_read"));
            EXPECT_TRUE(j["timings"].contains("PCI_write"));
            EXPECT_TRUE(j["timings"].contains("Scale"));
            EXPECT_TRUE(j["timings"].contains("Triad"));
        }
        EXPECT_TRUE(j.contains("results"));
        if (j.contains("results")) {
            EXPECT_TRUE(j["results"].contains("Add_avg_t"));
            EXPECT_TRUE(j["results"].contains("Add_best_rate"));
            EXPECT_TRUE(j["results"].contains("Add_max_t"));
            EXPECT_TRUE(j["results"].contains("Add_min_t"));
            EXPECT_TRUE(j["results"].contains("Copy_avg_t"));
            EXPECT_TRUE(j["results"].contains("Copy_best_rate"));
            EXPECT_TRUE(j["results"].contains("Copy_max_t"));
            EXPECT_TRUE(j["results"].contains("Copy_min_t"));
            EXPECT_TRUE(j["results"].contains("PCI_read_avg_t"));
            EXPECT_TRUE(j["results"].contains("PCI_read_best_rate"));
            EXPECT_TRUE(j["results"].contains("PCI_read_max_t"));
            EXPECT_TRUE(j["results"].contains("PCI_read_min_t"));
            EXPECT_TRUE(j["results"].contains("PCI_write_avg_t"));
            EXPECT_TRUE(j["results"].contains("PCI_write_best_rate"));
            EXPECT_TRUE(j["results"].contains("PCI_write_max_t"));
            EXPECT_TRUE(j["results"].contains("PCI_write_min_t"));
            EXPECT_TRUE(j["results"].contains("Scale_avg_t"));
            EXPECT_TRUE(j["results"].contains("Scale_best_rate"));
            EXPECT_TRUE(j["results"].contains("Scale_max_t"));
            EXPECT_TRUE(j["results"].contains("Scale_min_t"));
            EXPECT_TRUE(j["results"].contains("Triad_avg_t"));
            EXPECT_TRUE(j["results"].contains("Triad_best_rate"));
            EXPECT_TRUE(j["results"].contains("Triad_max_t"));
            EXPECT_TRUE(j["results"].contains("Triad_min_t"));
        }
    }
}
