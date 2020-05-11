//
// Created by Marius Meyer on 04.12.19
//
#include "gtest/gtest.h"
#include "setup/fpga_setup.hpp"
#include "parameters.h"

/**
 * Check if it is possible to find the platform and device that are given as default
 */
TEST (FPGASetup, FindValidPlatformAndDevice) {
    EXPECT_EQ (1, fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE).size());
}

/**
 * Checks if non existing platform leads to an error
 */
TEST (FPGASetup, FindNonExistingPlatform) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    // TODO regex does not work so for now its not tested!
    EXPECT_EXIT(fpga_setup::selectFPGADevice(DEFAULT_PLATFORM + 100, DEFAULT_DEVICE),
                ::testing::ExitedWithCode(1),
                ::testing::MatchesRegex(".*?Default platform \\d+ can not be used. Available platforms: \\d+"));
}

/**
 * Checks if non existing device leads to an error
 */
TEST (FPGASetup, FindNonExistingDevice) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    // TODO regex does not work so for now its not tested!
    EXPECT_EXIT(fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE + 100),
                ::testing::ExitedWithCode(1),
                ::testing::MatchesRegex(".*?Default device \\d+ can not be used. Available devices: \\d+"));
}


