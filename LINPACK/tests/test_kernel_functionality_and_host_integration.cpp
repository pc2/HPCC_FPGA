//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"
#include "nlohmann/json.hpp"

#ifdef _LAPACK_
#ifdef _DP
extern "C" void dgesv_(int* size, int* lrhs, double* A, int* size2, int* ipvt, double* b, int* size3, int* info);
#else
extern "C" void sgesv_(int* size, int* lrhs, float* A, int* size2, int* ipvt, float* b, int* size3, int* info);
#endif
#endif

struct LinpackKernelTest : testing::TestWithParam<uint> {
    
    std::unique_ptr<linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>> bm;
    std::unique_ptr<linpack::LinpackData<cl::Context>> data;
    uint array_size = 0;

    void SetUp() override {
        uint matrix_blocks = GetParam();
        bm = std::unique_ptr<linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>>(new linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->matrixSize = matrix_blocks * (1 << LOCAL_MEM_BLOCK_LOG);
        data = bm->generateInputData();
        array_size = bm->getExecutionSettings().programSettings->matrixSize;
    }

    void TearDown() override {
        bm = nullptr;
        data = nullptr;
    }

};


/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelTest, FPGACorrectResultsOneRepetition) {
    bm->executeKernel(*data);
    for (int i = 0; i < array_size; i++) {
        EXPECT_NEAR(data->b[i], 1.0, 1.0e-3);
    }
}

/**
 * GEFA Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelTest, DISABLED_FPGACorrectResultsGEFA) {
    bm->executeKernel(*data);
    auto data2 = bm->generateInputData();
    if (bm->getExecutionSettings().programSettings->isDiagonallyDominant) {
        linpack::gefa_ref_nopvt(data2->A, array_size, array_size);
    }
    else {
        linpack::gefa_ref(data2->A, array_size, array_size, data2->ipvt);
    }
    int errors = 0;
    for (int i = 0; i < array_size; i++) {
        for (int j = 0; j < array_size; j++) {
            if (std::fabs(data->A[i * array_size + j] - data2->A[i * array_size + j]) > 1.0e-3) {
                // Diagonal elements might be stored as the negative inverse to speed up calculation
                if ((std::fabs(data->A[i * array_size + j] - -1.0/ data2->A[i * array_size + j]) > 1.0e-3) || i != j) {
                    errors++;
                }
            }
        }
    }
    EXPECT_EQ(0, errors);
}

#ifdef _LAPACK_
/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelTest, DISABLED_ValidationWorksForMKL) {

    int info;    
    auto data_cpu = bm->generateInputData();
    int s = static_cast<int>(array_size);
    int lrhs = 1;
#ifndef _DP
        sgesv_(&s, &lrhs, data_cpu->A, &s, data_cpu->ipvt, data_cpu->b, &s, &info);
#else
        dgesv_(&s, &lrhs, data_cpu->A, &s, data_cpu->ipvt, data_cpu->b, &s, &info);
#endif
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError(); 
}


#endif

using json = nlohmann::json;

TEST_P(LinpackKernelTest, JsonDump) {
    bm->executeKernel(*data);
    bm->collectResults();
    bm->dumpConfigurationAndResults("linpack.json");
    std::FILE *f = std::fopen("linpack.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        json j = json::parse(f);
        EXPECT_TRUE(j.contains("timings"));
        if (j.contains("timings")) {
            EXPECT_TRUE(j["timings"].contains("gefa"));
            EXPECT_TRUE(j["timings"].contains("gesl"));
        }
        EXPECT_TRUE(j.contains("results"));
        if (j.contains("results")) {
            EXPECT_TRUE(j["results"].contains("gflops"));
            EXPECT_TRUE(j["results"].contains("gflops_lu"));
            EXPECT_TRUE(j["results"].contains("gflops_sl"));
            EXPECT_TRUE(j["results"].contains("t_mean"));
            EXPECT_TRUE(j["results"].contains("t_min"));
            EXPECT_TRUE(j["results"].contains("tlu_mean"));
            EXPECT_TRUE(j["results"].contains("tlu_min"));
            EXPECT_TRUE(j["results"].contains("tsl_mean"));
            EXPECT_TRUE(j["results"].contains("tsl_min"));
        }
    }
}

INSTANTIATE_TEST_CASE_P(
        LinpackKernelParametrizedTests,
        LinpackKernelTest,
        ::testing::Values(1, 2, 3));
