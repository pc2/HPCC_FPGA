

#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"

#ifdef _LAPACK_
#ifdef _DP
extern "C" void dgesv_(int* size, int* lrhs, double* A, int* size2, int* ipvt, double* b, int* size3, int* info);
#else
extern "C" void sgesv_(int* size, int* lrhs, float* A, int* size2, int* ipvt, float* b, int* size3, int* info);
#endif
#endif


struct LinpackHostTest : testing::Test {
    
    std::unique_ptr<linpack::LinpackBenchmark> bm;
    std::unique_ptr<linpack::LinpackData> data;
    int array_size = 0;

    void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->matrixSize = 1 << LOCAL_MEM_BLOCK_LOG;
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        data = bm->generateInputData();
        array_size = bm->getExecutionSettings().programSettings->matrixSize;
    }

    void TearDown() override {
        bm = nullptr;
        data = nullptr;
    }

};

TEST_F(LinpackHostTest, GenerateUniformMatrixWorksCorrectly) {
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    data = bm->generateInputData();
    for (int i=0; i < array_size; i++) {
        for (int j=0; j < array_size; j++) {
            EXPECT_TRUE((data->A[array_size * i + j] > 0.0) && (data->A[array_size * j + i] < 1.0));
        }
    }
}

TEST_F(LinpackHostTest, GenerateDiagonallyDominantMatrixWorksCorrectly) {
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
    data = bm->generateInputData();
    for (int i=0; i < array_size; i++) {
        HOST_DATA_TYPE sum = 0;
        for (int j=0; j < array_size; j++) {
            if (i != j) {
                EXPECT_TRUE((data->A[array_size * i + j] > 0.0) && (data->A[array_size * i + j] < 1.0));
                sum += data->A[array_size * i + j];
            }
        }
        EXPECT_FLOAT_EQ(data->A[array_size*i + i], sum);
    }
}

TEST_F(LinpackHostTest, ReferenceSolveWithoutPivoting) {
    data = bm->generateInputData();
    linpack::gefa_ref_nopvt(data->A, array_size, array_size);
    linpack::gesl_ref_nopvt(data->A, data->b, array_size, array_size);
    EXPECT_TRUE(bm->validateOutputAndPrintError(*data));
}
