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
    }
};



/**
 * Check if the data generation works as expected
 */
TEST_F(TransposeHandlersTest, DataGenerationDistExtFailsForMPISizeEquals1) {
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,1);
    EXPECT_THROW(handler->generateData(bm->getExecutionSettings()), std::runtime_error);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtForOneReplication) {
    bm->getExecutionSettings().programSettings->kernelReplications = 1;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,2);
    auto data = handler->generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->A.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->B.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->result.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->numBlocks, 2);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtForTwoReplications) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 4;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,2);
    auto data = handler->generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->A.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->B.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->result.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->numBlocks, 4);
}

TEST_F(TransposeHandlersTest, DataGenerationWorksDistExtForOneReplicationTwoBlocks) {
    bm->getExecutionSettings().programSettings->kernelReplications = 2;
    bm->getExecutionSettings().programSettings->matrixSize = bm->getExecutionSettings().programSettings->blockSize * 2 * bm->getExecutionSettings().programSettings->kernelReplications * 2;
    auto handler = transpose::dataHandlerIdentifierMap[TRANSPOSE_HANDLERS_DIST_EXT](0,2);
    auto data = handler->generateData(bm->getExecutionSettings());
    EXPECT_EQ(data->A.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->B.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->result.size(), bm->getExecutionSettings().programSettings->kernelReplications);
    EXPECT_EQ(data->numBlocks, 16);
}
