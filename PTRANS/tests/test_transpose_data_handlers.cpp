//
// Created by Marius Meyer on 19.10.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "CL/cl.hpp"
#include "test_program_settings.h"
#include "gmock/gmock-matchers.h"
#include "transpose_benchmark.hpp"
#include "data_handlers/diagonal.hpp"


struct TransposeHandlersTest : testing::Test {
    std::unique_ptr<transpose::TransposeBenchmark> bm;

    TransposeHandlersTest() {
        bm = std::unique_ptr<transpose::TransposeBenchmark>( new transpose::TransposeBenchmark(global_argc, global_argv));
    }

    void SetUp() override {
        bm->getExecutionSettings().programSettings->blockSize = 4;
        bm->getExecutionSettings().programSettings->matrixSize = 12;
    }
};

// TODO check data exchange and validation based on exchanged data

/**
 * Test DitExt class instantiation
 */
TEST_F(TransposeHandlersTest, DistDiagCreateHandlerSuccess) {
    EXPECT_NO_THROW(transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1));
}

TEST_F(TransposeHandlersTest, DistDiagCreateHandlerFail) {
    EXPECT_THROW(transpose::data_handler::DistributedDiagonalTransposeDataHandler(1,1), std::runtime_error);
}

/**
 * Test, if the total number of generated blocks matches the total matrix size
 * 
 */
TEST_F(TransposeHandlersTest, DistDiagNumberOfBlocksCorrectForMPI1Block1) {
    uint mpi_size = 1;
    uint matrix_size_in_blocks = 1;

    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4* matrix_size_in_blocks;
    uint block_count = 0;
    for (int i=0; i < mpi_size; i++) {
        auto h = transpose::data_handler::DistributedDiagonalTransposeDataHandler(i, mpi_size);
        auto d = h.generateData(bm->getExecutionSettings());
        block_count += d->numBlocks;
    }
    EXPECT_EQ(block_count, matrix_size_in_blocks * matrix_size_in_blocks);
}

TEST_F(TransposeHandlersTest, DistDiagNumberOfBlocksCorrectForMPI3Block3) {
    uint mpi_size = 3;
    uint matrix_size_in_blocks = 3;

    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4* matrix_size_in_blocks;
    uint block_count = 0;
    for (int i=0; i < mpi_size; i++) {
        auto h = transpose::data_handler::DistributedDiagonalTransposeDataHandler(i, mpi_size);
        auto d = h.generateData(bm->getExecutionSettings());
        block_count += d->numBlocks;
    }
    EXPECT_EQ(block_count, matrix_size_in_blocks * matrix_size_in_blocks);
}

TEST_F(TransposeHandlersTest, DistDiagNumberOfBlocksCorrectForMPI9Block3) {
    uint mpi_size = 9;
    uint matrix_size_in_blocks = 3;

    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4* matrix_size_in_blocks;
    uint block_count = 0;
    for (int i=0; i < mpi_size; i++) {
        auto h = transpose::data_handler::DistributedDiagonalTransposeDataHandler(i, mpi_size);
        auto d = h.generateData(bm->getExecutionSettings());
        block_count += d->numBlocks;
    }
    EXPECT_EQ(block_count, matrix_size_in_blocks * matrix_size_in_blocks);
}

TEST_F(TransposeHandlersTest, DistDiagNumberOfBlocksCorrectForMPI5Block4) {
    uint mpi_size = 5;
    uint matrix_size_in_blocks = 4;

    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4* matrix_size_in_blocks;
    uint block_count = 0;
    for (int i=0; i < mpi_size; i++) {
        auto h = transpose::data_handler::DistributedDiagonalTransposeDataHandler(i, mpi_size);
        auto d = h.generateData(bm->getExecutionSettings());
        block_count += d->numBlocks;
    }
    EXPECT_EQ(block_count, matrix_size_in_blocks * matrix_size_in_blocks);
}

/**
 * @brief Test data generation for DistDiag
 * 
 */
TEST_F(TransposeHandlersTest, DataGenerationDistDiagSucceedsForMPISizeEquals1SingleBlock) {
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    EXPECT_NO_THROW(handler.generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistDiagSucceedsForMPISizeEquals1Blocks9) {
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4*3;
    EXPECT_THROW(handler.generateData(bm->getExecutionSettings()), std::runtime_error);
}

TEST_F(TransposeHandlersTest, DataGenerationDistDiagSucceedsForMPISizeEquals3Blocks9) {
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4*3;
    EXPECT_NO_THROW(handler.generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistDiagFailsForMPISizeEquals3Blocks1) {
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    EXPECT_NO_THROW(handler.generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistDiagFailsForMPISizeEquals3Blocks4) {
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4 * 2;
    EXPECT_THROW(handler.generateData(bm->getExecutionSettings()), std::runtime_error);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistDiagForOneReplication) {
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize;
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    auto data = handler.generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->blockSize, bm->getExecutionSettings().programSettings->blockSize);
    EXPECT_EQ(data->numBlocks, 1);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistDiagForTwoReplications) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize;
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    auto data = handler.generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->blockSize, bm->getExecutionSettings().programSettings->blockSize);
    EXPECT_EQ(data->numBlocks, 1);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistDiagReproducableA) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize;
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    auto data = handler.generateData(bm->getExecutionSettings());
    auto data2 = handler.generateData(bm->getExecutionSettings());
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->A[i] - data2->A[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistDiagReproducableB) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize;
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    auto data = handler.generateData(bm->getExecutionSettings());
    auto data2 = handler.generateData(bm->getExecutionSettings());
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->B[i] - data2->B[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistDiagExchangeWorksForSingleRank) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize;
    auto handler = transpose::data_handler::DistributedDiagonalTransposeDataHandler(0,1);
    auto data = handler.generateData(bm->getExecutionSettings());
    auto data2 = handler.generateData(bm->getExecutionSettings());
    handler.exchangeData(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->A[i] - data2->A[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}


/**
 * Check if the generated input data is in the specified range
 */
TEST_F(TransposeHandlersTest, GenerateInputDataInRangeDistDiagSingleBlock) {
    bm->getExecutionSettings().programSettings->matrixSize = 5;
    bm->getExecutionSettings().programSettings->blockSize = 5;
    auto data = bm->generateInputData();
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_LT(data->A[i * 5 + j], 100);
            EXPECT_GT(data->A[i * 5 + j], -100);
            EXPECT_LT(data->B[i * 5 + j], 101);
            EXPECT_GT(data->B[i * 5 + j], -99);
        }
    }
}

/**
 * Check if the generated input data is in the specified range
 */
TEST_F(TransposeHandlersTest, GenerateInputDataInRangeDistDiagMultipleBlocks) {
    bm->getExecutionSettings().programSettings->matrixSize = 6;
    bm->getExecutionSettings().programSettings->blockSize = 2;
    EXPECT_THROW(bm->generateInputData(), std::runtime_error);
}


