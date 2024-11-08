

#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"


struct LinpackHostTest : testing::Test {
    
    std::unique_ptr<linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>> bm;
    std::unique_ptr<linpack::LinpackData<cl::Context>> data;
    int array_size = 0;

    void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>>(new linpack::LinpackBenchmark<cl::Device, cl::Context, cl::Program>(global_argc, global_argv));
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

#ifndef _DP
TEST_F(LinpackHostTest, ReferenceSolveGMRES) {
    data = bm->generateInputData();
    auto A = std::unique_ptr<double[]>(new double[array_size * array_size]);
    auto LU = std::unique_ptr<double[]>(new double[array_size * array_size]);
    auto x = std::unique_ptr<double[]>(new double[array_size]);
    auto b = std::unique_ptr<double[]>(new double[array_size]);
    // convert generated matrix to double
    for (int i=0; i < array_size; i++) {
        for (int j=0; j < array_size; j++) {
            A[i * array_size + j] = static_cast<double>(data->A[i * array_size + j]);
            LU[i * array_size + j] = static_cast<double>(data->A[i * array_size + j]);
        }
        b[i] = static_cast<double>(data->b[i]);
        x[i] = static_cast<double>(data->b[i]);
    }
    double tol = std::numeric_limits<double>::epsilon() / 2.0 / (array_size / 4.0);
    gmres_ref(array_size, A.get(),array_size, x.get(), b.get(), LU.get(),array_size,50,1,0.00000001);
    // convert result vector to float
    for (int i=0; i < array_size; i++) {
        data->b[i] = static_cast<float>(x[i]);
    }
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError(); 
}
#endif

TEST_F(LinpackHostTest, ReferenceSolveWithPivoting) {
    bm->getExecutionSettings().programSettings->isDiagonallyDominant = false;
    data = bm->generateInputData();
    linpack::gefa_ref(data->A, array_size, array_size, data->ipvt);
    linpack::gesl_ref(data->A, data->b, data->ipvt, array_size, array_size);
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError(); 
}


TEST_F(LinpackHostTest, ReferenceSolveWithoutPivoting) {
    data = bm->generateInputData();
    linpack::gefa_ref_nopvt(data->A, array_size, array_size);
    linpack::gesl_ref_nopvt(data->A, data->b, array_size, array_size);
    EXPECT_TRUE(bm->validateOutput(*data));
    bm->printError(); 
}


