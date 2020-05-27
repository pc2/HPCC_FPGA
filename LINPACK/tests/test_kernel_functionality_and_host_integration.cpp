//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#ifdef _INTEL_MKL_
#include "mkl.h"
#endif

struct LinpackKernelTest : testing::Test {
    
    std::unique_ptr<linpack::LinpackBenchmark> bm;
    std::unique_ptr<linpack::LinpackData> data;
    uint array_size = 0;

    void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->matrixSize = 1 << LOCAL_MEM_BLOCK_LOG;
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
TEST_F(LinpackKernelTest, FPGACorrectResultsOneRepetition) {

    auto result = bm->executeKernel(*data);
    for (int i = 0; i < array_size; i++) {
        EXPECT_NEAR(data->b[i], 1.0, 1.0e-3);
    }
}

#ifdef __INTEL_MKL__
/**
 * Execution returns correct results for a single repetition
 */
TEST_F(LinpackKernelTest, ValidationWorksForMKL) {

    int info;    
    auto data_cpu = bm->generateInputData();
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            data_cpu->A[i * array_size + j] = data->A[j* array_size + i];
        }
    }
    int s = static_cast<int>(array_size);
    int lrhs = 1;
#ifndef _DP
        sgesv(&s, &lrhs, data_cpu->A, &s, data_cpu->ipvt, data_cpu->b, &s, &info);
#else
        dgesv(&s, &lrhs, data_cpu->A, &s, data_cpu->ipvt, data_cpu->b, &s, &info);
#endif
    bool success = bm->validateOutputAndPrintError(*data_cpu);
    EXPECT_TRUE(success);
}


#endif
