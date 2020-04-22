//
// Created by Marius Meyer on 04.12.19
//
#include "gtest/gtest.h"
#include "../src/host/setup/fpga_setup.hpp"
#include "parameters.h"
#include "mpi.h"

/**
 * Check if it is possible to find the platform and device that are given as default
 */
TEST (FPGASetup, FindValidPlatformAndDevice) {
    MPI_Init(NULL, NULL);
    EXPECT_EQ (1, fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE).size());
}

/**
 * Checks if non existing platform leads to an error
 */
TEST (FPGASetup, FindNonExistingPlatform) {
    MPI_Init(NULL, NULL);
    // TODO regex does not work so for now its not tested!
    EXPECT_EXIT(fpga_setup::selectFPGADevice(DEFAULT_PLATFORM + 100, DEFAULT_DEVICE),
                ::testing::ExitedWithCode(1),
                ::testing::MatchesRegex(".*"));
}

/**
 * Checks if non existing device leads to an error
 */
TEST (FPGASetup, FindNonExistingDevice) {
    MPI_Init(NULL, NULL);
    // TODO regex does not work so for now its not tested!
    EXPECT_EXIT(fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE + 100),
                ::testing::ExitedWithCode(1),
                ::testing::MatchesRegex(".*"));
}

