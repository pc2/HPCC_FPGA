//
// Created by Marius Meyer on 04.12.19
//


#include "gtest/gtest.h"
#include "setup/fpga_setup.hpp"
#include "parameters.h"
#include "test_program_settings.h"
#include "gmock/gmock.h"


/**
 * Checks if non existing platform leads to an error
 */
TEST (FPGASetup, FindNonExistingPlatform) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    std::stringstream fmt;
    fmt << "Default platform " << bm->getExecutionSettings().programSettings->defaultPlatform + 100 << " can not be used. Available platforms: " ;
    EXPECT_EXIT(fpga_setup::selectFPGADevice(bm->getExecutionSettings().programSettings->defaultPlatform + 100, bm->getExecutionSettings().programSettings->defaultDevice),
                ::testing::ExitedWithCode(1),
                ::testing::StartsWith(fmt.str()));
}

/**
 * Checks if non existing device leads to an error
 */
TEST (FPGASetup, FindNonExistingDevice) {
    testing::FLAGS_gtest_death_test_style="threadsafe";
    std::stringstream fmt;
    fmt << "Default device " << bm->getExecutionSettings().programSettings->defaultDevice + 100 << " can not be used. Available devices: " ;
    EXPECT_EXIT(fpga_setup::selectFPGADevice(bm->getExecutionSettings().programSettings->defaultPlatform, bm->getExecutionSettings().programSettings->defaultDevice + 100),
                ::testing::ExitedWithCode(1),
                ::testing::StartsWith(fmt.str()));
}
