//
// Created by Marius Meyer on 04.12.19
//
#include "gtest/gtest.h"
#include "../src/host/random_access_functionality.hpp"
#include "parameters.h"

/**
 * Check if the correctness test gives correct results for correct array
 */
TEST (FPGASetup, ResultValidationWorksForCorrectUpdates) {
    HOST_DATA_TYPE data[1024];
    generateInputData(data, 1024);
    // do random accesses
    checkRandomAccessResults(data, 1024);
    // check correctness of random accesses
    double error = checkRandomAccessResults(data, 1024);
    ASSERT_FLOAT_EQ(error, 0.0);
}


/**
 * Check if the correctness test gives correct results for not updated array
 */
TEST (FPGASetup, ResultValidationWorksForWrongUpdates) {
    HOST_DATA_TYPE data[1024];
    generateInputData(data, 1024);
    // check correctness of random accesses
    double error = checkRandomAccessResults(data, 1024);
    ASSERT_GT(error, 0.3);
}
