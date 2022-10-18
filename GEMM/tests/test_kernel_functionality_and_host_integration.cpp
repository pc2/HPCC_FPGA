//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "gemm_benchmark.hpp"
#include "parameters.h"
#include "test_program_settings.h"

void
ref_matmul(HOST_DATA_TYPE* A, HOST_DATA_TYPE* B, HOST_DATA_TYPE* C, int size) {
    for (int i=0; i< size; i++) {
        for (int j=0; j< size; j++) {
            C[i * size + j] = OPTIONAL_CAST(0.0);
        }
    }
    gemm::gemm_ref(A,B,C,size,OPTIONAL_CAST(1.0),OPTIONAL_CAST(0.0));
}


struct GEMMKernelTest : testing::Test, testing::WithParamInterface<unsigned> {
    std::unique_ptr<gemm::GEMMBenchmark> bm;
    std::unique_ptr<gemm::GEMMData> data;
    unsigned matrix_size;

    GEMMKernelTest() {
        bm = std::unique_ptr<gemm::GEMMBenchmark>(new gemm::GEMMBenchmark(global_argc, global_argv));
        matrix_size = GetParam() * BLOCK_SIZE;
        bm->getExecutionSettings().programSettings->matrixSize = matrix_size;
    }

    void SetUp() {
        data = bm->generateInputData();
    }
};

/**
 * Tests if C will be multiplied by beta
 */
TEST_P(GEMMKernelTest, FPGACorrectNumberOfRepetitionsIs1) {
    bm->getExecutionSettings().programSettings->numRepetitions = 1;
    bm->executeKernel(*data);
    EXPECT_EQ(bm->getTimingsMap().at("execution").size(), 1);
}

/**
 * Tests if C will be multiplied by beta
 */
TEST_P(GEMMKernelTest, FPGACorrectNumberOfRepetitionsIs3) {
    bm->getExecutionSettings().programSettings->numRepetitions = 3;
    bm->executeKernel(*data);
    EXPECT_EQ(bm->getTimingsMap().at("execution").size(), 3);
}

/**
 * Tests if C will be multiplied by beta
 */
TEST_P(GEMMKernelTest, FPGACorrectCtimesBeta) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = OPTIONAL_CAST(0.0);
            data->B[i * matrix_size + j] = OPTIONAL_CAST(0.0);
            data->C[i * matrix_size + j] = OPTIONAL_CAST(1.0);
        }
    }
    bm->executeKernel(*data);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(data->C_out[i * matrix_size + j], 2.0 * data->C[i * matrix_size + j], std::numeric_limits<HOST_DATA_TYPE>::epsilon());
        }
    }
}

/**
 * Tests if A will be multiplied by alpha
 */
TEST_P(GEMMKernelTest, FPGACorrectAtimesAlpha) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->B[i * matrix_size + j] = i == j ? OPTIONAL_CAST(1.0) : OPTIONAL_CAST(0.0);
            data->C[i * matrix_size + j] = OPTIONAL_CAST(0.0);
        }
    }
    data->alpha = 2.0;
    data->beta = 0.0;

    bm->executeKernel(*data);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(data->C_out[i * matrix_size + j], 2.0 * data->A[i * matrix_size + j], std::numeric_limits<HOST_DATA_TYPE>::epsilon());
        }
    }
}

/**
 * Tests if B will be multiplied by alpha
 */
TEST_P(GEMMKernelTest, FPGACorrectBtimesAlpha) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->A[i * matrix_size + j] = i == j ? OPTIONAL_CAST(1.0) : OPTIONAL_CAST(0.0);
            data->C[i * matrix_size + j] = OPTIONAL_CAST(0.0);
        }
    }
    data->alpha = 2.0;
    data->beta = 0.0;
    bm->executeKernel(*data);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(data->C_out[i * matrix_size + j], 2.0 * data->B[i * matrix_size + j], std::numeric_limits<HOST_DATA_TYPE>::epsilon());
        }
    }
}

/**
 * Tests if A will be multiplied with B
 */
TEST_P(GEMMKernelTest, FPGACorrectAmulB) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->C[i * matrix_size + j] = OPTIONAL_CAST(0.0);
            data->A[i * matrix_size + j] = OPTIONAL_CAST(j % 10);
            data->B[i * matrix_size + j] = OPTIONAL_CAST(i % 10);
        }
    }
    data->alpha = 1.0;
    data->beta = 1.0;
    bm->executeKernel(*data);

    HOST_DATA_TYPE c_ref_out[matrix_size * matrix_size];
    ref_matmul(data->A,data->B,c_ref_out,matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(data->C_out[i * matrix_size + j], c_ref_out[i * matrix_size + j], 0.001);
        }
    }
}

/**
 * Tests if C will be added to A
 */
TEST_P(GEMMKernelTest, FPGACorrectCplusA) {

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            data->B[i * matrix_size + j] = i == j ? 1.0 : 0.0;
        }
    }
    data->alpha = 1.0;
    data->beta = 1.0;

    bm->executeKernel(*data);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_FLOAT_EQ(data->C_out[i * matrix_size + j], data->A[i * matrix_size + j] + data->C[i * matrix_size + j]);
        }
    }
}


/**
 * Tests full multiply add
 */

TEST_P(GEMMKernelTest, FPGACorrectbetaCplusalphaAB) {
    HOST_DATA_TYPE c_ref_out[matrix_size * matrix_size];
    bm->executeKernel(*data);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
           c_ref_out[i * matrix_size + j] = data->C[i * matrix_size + j];
        }
    }
    gemm::gemm_ref(data->A,data->B,c_ref_out,matrix_size,OPTIONAL_CAST(0.5),OPTIONAL_CAST(2.0));
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(data->C_out[i * matrix_size + j], c_ref_out[i * matrix_size + j], std::numeric_limits<HOST_DATA_TYPE>::epsilon() * matrix_size * matrix_size);
        }
    }
}

INSTANTIATE_TEST_CASE_P(Default, GEMMKernelTest,
         testing::Values(1,2));

