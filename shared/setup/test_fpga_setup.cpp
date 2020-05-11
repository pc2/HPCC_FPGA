//
// Created by Marius Meyer on 04.12.19
//


#include "gtest/gtest.h"
#include "setup/fpga_setup.hpp"
#include "parameters.h"
#include "testing/test_program_settings.h"
#include "gmock/gmock.h"


/**
 * Check if it is possible to find the platform and device that are given as default
 */
TEST (FPGASetup, FindValidPlatformAndDevice) {
    EXPECT_EQ (1, fpga_setup::selectFPGADevice(programSettings->defaultPlatform, programSettings->defaultDevice).size());
}

/**
 * Checks if non existing platform leads to an error
 */
TEST (FPGASetup, FindNonExistingPlatform) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    std::stringstream fmt;
    fmt << "Default platform " << programSettings->defaultPlatform + 100 << " can not be used. Available platforms: " ;
    EXPECT_EXIT(fpga_setup::selectFPGADevice(programSettings->defaultPlatform + 100, programSettings->defaultDevice),
                ::testing::ExitedWithCode(1),
                ::testing::StartsWith(fmt.str()));
}

/**
 * Checks if non existing device leads to an error
 */
TEST (FPGASetup, FindNonExistingDevice) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    std::stringstream fmt;
    fmt << "Default device " << programSettings->defaultDevice + 100 << " can not be used. Available devices: " ;
    EXPECT_EXIT(fpga_setup::selectFPGADevice(programSettings->defaultPlatform, programSettings->defaultDevice + 100),
                ::testing::ExitedWithCode(1),
                ::testing::StartsWith(fmt.str()));
}
