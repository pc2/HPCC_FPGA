//
// Created by Marius Meyer on 04.12.19
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "random_access_benchmark.hpp"
#include "test_program_settings.h"


struct RandomAccessHostCodeTest : testing::Test {

    RandomAccessHostCodeTest() {
        bm->getExecutionSettings().programSettings->dataSize = 1024;
    }

};

/**
 * Check if the correctness test gives correct results for correct array
 */
TEST_F(RandomAccessHostCodeTest, ResultValidationWorksForCorrectUpdates) {
    auto data = bm->generateInputData();
    // do random accesses
    bm->validateOutputAndPrintError(*data);
    // check correctness of random accesses
    bool success = bm->validateOutputAndPrintError(*data);
    EXPECT_TRUE(success);
}


/**
 * Check if the correctness test gives correct results for not updated array
 */
TEST_F(RandomAccessHostCodeTest, ResultValidationWorksForWrongUpdates) {
    auto data = bm->generateInputData();
    // check correctness of random accesses
    bool success = bm->validateOutputAndPrintError( *data);
    EXPECT_FALSE(success);
}
