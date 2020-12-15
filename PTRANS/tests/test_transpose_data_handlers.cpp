//
// Created by Marius Meyer on 19.10.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "CL/cl.hpp"
#include "test_program_settings.h"
#include "gmock/gmock-matchers.h"
#include "transpose_benchmark.hpp"
#include "transpose_handlers.hpp"


struct TransposeHandlersTest : testing::Test {
    std::unique_ptr<transpose::TransposeBenchmark> bm;

    TransposeHandlersTest() {
        bm = std::unique_ptr<transpose::TransposeBenchmark>( new transpose::TransposeBenchmark(global_argc, global_argv));
    }

    void SetUp() override {
        bm->getExecutionSettings().programSettings->blockSize = 4;
        bm->getExecutionSettings().programSettings->matrixSize = 4;
    }
};

// TODO check data exchange and validation based on exchanged data



/**
 * Test DitExt class instantiation
 */
TEST_F(TransposeHandlersTest, InstantiationDistExtFailsForMPISizeEquals2) {
    EXPECT_THROW(transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,2), std::runtime_error);
}

TEST_F(TransposeHandlersTest, InstantiationDistExtFailsForMPISizeEquals4) {
    EXPECT_THROW(transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,4), std::runtime_error);
}

TEST_F(TransposeHandlersTest, InstantiationDistExtSucceedsForMPISizeEquals3) {
    EXPECT_NO_THROW(transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,3));
}

TEST_F(TransposeHandlersTest, InstantiationDistExtSucceedsForMPISizeEquals1) {
    EXPECT_NO_THROW(transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1));
}

/**
 * @brief Test data generation for DistExt
 * 
 */
TEST_F(TransposeHandlersTest, DataGenerationDistExtSucceedsForMPISizeEquals1SingleBlock) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    EXPECT_NO_THROW(handler->generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistExtSucceedsForMPISizeEquals1Blocks9) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4*3;
    EXPECT_NO_THROW(handler->generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistExtSucceedsForMPISizeEquals3Blocks9) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4*3;
    EXPECT_NO_THROW(handler->generateData(bm->getExecutionSettings()));
}

TEST_F(TransposeHandlersTest, DataGenerationDistExtFailsForMPISizeEquals3Blocks1) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4;
    EXPECT_THROW(handler->generateData(bm->getExecutionSettings()), std::runtime_error);
}

TEST_F(TransposeHandlersTest, DataGenerationDistExtFailsForMPISizeEquals3Blocks4) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,3);
    bm->getExecutionSettings().programSettings->blockSize = 4;
    bm->getExecutionSettings().programSettings->matrixSize = 4 * 2;
    EXPECT_THROW(handler->generateData(bm->getExecutionSettings()), std::runtime_error);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtForOneReplication) {
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    auto data = handler->generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->blockSize, bm->getExecutionSettings().programSettings->blockSize);
    EXPECT_EQ(data->numBlocks, 4);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtForTwoReplications) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    auto data = handler->generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->blockSize, bm->getExecutionSettings().programSettings->blockSize);
    EXPECT_EQ(data->numBlocks, 4);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtReproducableA) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    auto data = handler->generateData(bm->getExecutionSettings());
    auto data2 = handler->generateData(bm->getExecutionSettings());
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->A[i] - data2->A[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtReproducableB) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    auto data = handler->generateData(bm->getExecutionSettings());
    auto data2 = handler->generateData(bm->getExecutionSettings());
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->B[i] - data2->B[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtExchangeWorksForSingleRank) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    auto data = handler->generateData(bm->getExecutionSettings());
    auto data2 = handler->generateData(bm->getExecutionSettings());
    handler->exchangeData(*data);
    double aggregated_error = 0.0;
    for (int i = 0; i < data->blockSize * data->blockSize * data->numBlocks; i++) {
        aggregated_error += std::fabs(data->A[i] - data2->A[i]);
    }
    EXPECT_FLOAT_EQ(aggregated_error, 0.0);
}
