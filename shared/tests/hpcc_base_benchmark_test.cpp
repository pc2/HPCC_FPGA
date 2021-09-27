//
// Created by Marius Meyer on 04.12.19
//


#include "gtest/gtest.h"
#include "setup/fpga_setup.hpp"
#include "test_program_settings.h"
#include "gmock/gmock.h"
#include "hpcc_benchmark.hpp"


// Dirty GoogleTest and static library hack
// Use this function in your test main
// This will make the linker link this library 
// and enable the included tests
void use_hpcc_base_lib() {}

class MinimalBenchmark : public hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, int, int> {

protected:

    void
    addAdditionalParseOptions(cxxopts::Options &options) override {
        options.allow_unrecognised_options();
    }

public:

    bool returnInputData = true;  
    bool returnExecuteKernel = true; 
    bool returnValidate = true;
    bool configurationCheckSucceeds = true;

    std::unique_ptr<int>
    generateInputData() override { return returnInputData ? std::unique_ptr<int>(new int) : std::unique_ptr<int>(nullptr);}

    std::unique_ptr<int>
    executeKernel(int &data) override { return returnExecuteKernel ? std::unique_ptr<int>(new int) : std::unique_ptr<int>(nullptr);}

    bool
    validateOutputAndPrintError(int &data) override { return returnValidate;}

    bool
    checkInputParameters() override { return configurationCheckSucceeds;}

    void
    collectAndPrintResults(const int &output) override {}

    MinimalBenchmark() : HpccFpgaBenchmark(0, { nullptr}) {}

};


class SuccessBenchmark : public hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, int, int> {

protected:

    void
    addAdditionalParseOptions(cxxopts::Options &options) override {
        options.allow_unrecognised_options();
    }

public:

    bool returnInputData = true;  
    bool returnExecuteKernel = true; 
    bool returnValidate = true;
    bool forceSetupFail = false;

    uint executeKernelcalled = 0;
    uint generateInputDatacalled = 0;
    uint validateOutputcalled = 0;

    std::unique_ptr<int>
    generateInputData() override { 
        if (!returnInputData) {
            throw fpga_setup::FpgaSetupException("Test input data failed");
        }
        generateInputDatacalled++;
        return std::unique_ptr<int>(new int);}

    std::unique_ptr<int>
    executeKernel(int &data) override { 
        if (!returnExecuteKernel) {
            throw fpga_setup::FpgaSetupException("Test execute kernel failed");
        }
        executeKernelcalled++;
        return std::unique_ptr<int>(new int);}

    bool
    validateOutputAndPrintError(int &data) override { 
        validateOutputcalled++;
        return returnValidate;}

    void
    collectAndPrintResults(const int &output) override {}

    bool
    checkInputParameters() override {
        if (forceSetupFail) {
            return false;
        }
        else {
            return hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, int, int>::checkInputParameters();
        }
    }

    SuccessBenchmark() : HpccFpgaBenchmark(0, { nullptr}) {}

};

class BaseHpccBenchmarkTest :public  ::testing::Test {

public:
    std::unique_ptr<SuccessBenchmark> bm;

    BaseHpccBenchmarkTest() {
        bm = std::unique_ptr<SuccessBenchmark>(new SuccessBenchmark());
        bm->setupBenchmark(global_argc, global_argv);
    }

};


TEST_F(BaseHpccBenchmarkTest, SetupSucceedsForBenchmarkTest) {
        bool success = bm->setupBenchmark(global_argc, global_argv);
        EXPECT_TRUE(success);
}


/**
 * Checks if the testing flag works as expected
 */
TEST_F(BaseHpccBenchmarkTest, AllExecutedWhenNotTestOnly) {
    bm->getExecutionSettings().programSettings->testOnly = false;
    bm->executeBenchmark();
    EXPECT_EQ(bm->validateOutputcalled, 1);
    EXPECT_EQ(bm->executeKernelcalled, 1);
    EXPECT_EQ(bm->generateInputDatacalled, 1);
}

TEST_F(BaseHpccBenchmarkTest, NothingExecutedWhenTestOnly) {
    bm->getExecutionSettings().programSettings->testOnly = true;
    bm->executeBenchmark();
    EXPECT_EQ(bm->validateOutputcalled, 0);
    EXPECT_EQ(bm->executeKernelcalled, 0);
    EXPECT_EQ(bm->generateInputDatacalled, 0);
}

TEST_F(BaseHpccBenchmarkTest, ExecutionSuccessWhenNotTestOnly) {
    bm->getExecutionSettings().programSettings->testOnly = false;
    EXPECT_TRUE(bm->executeBenchmark());

}

TEST_F(BaseHpccBenchmarkTest, ExecutionFailsWhenTestOnlyAndSetupFails) {
    bm->getExecutionSettings().programSettings->testOnly = true;
    bm->forceSetupFail = true;
    bm->setupBenchmark(global_argc, global_argv);
    EXPECT_FALSE(bm->executeBenchmark());
}

TEST_F(BaseHpccBenchmarkTest, ExecutionSuccessWhenTestOnlyAndSetupSuccess) {
    bm->getExecutionSettings().programSettings->testOnly = true;
    EXPECT_TRUE(bm->executeBenchmark());
}

/**
 * Checks if using default platform and device is successful
 */
TEST_F(BaseHpccBenchmarkTest, SuccessUseDefaultPlatform) {
    EXPECT_NE(fpga_setup::selectFPGADevice(bm->getExecutionSettings().programSettings->defaultPlatform, bm->getExecutionSettings().programSettings->defaultDevice).get(), nullptr);
}

/**
 * Checks if non existing platform leads to an error
 */
TEST_F(BaseHpccBenchmarkTest, FindNonExistingPlatform) {
    ASSERT_THROW(fpga_setup::selectFPGADevice(100, bm->getExecutionSettings().programSettings->defaultDevice).get(), fpga_setup::FpgaSetupException);
}

/**
 * Checks if non existing device leads to an error
 */
TEST_F(BaseHpccBenchmarkTest, FindNonExistingDevice) {
    ASSERT_THROW(fpga_setup::selectFPGADevice(bm->getExecutionSettings().programSettings->defaultPlatform, 100).get(), fpga_setup::FpgaSetupException);
}

/**
 * Execute kernel and validation is success
 */
TEST_F(BaseHpccBenchmarkTest, SuccessfulExeAndVal) {
    EXPECT_TRUE(bm->executeBenchmark());
}

/**
 * Execute kernel is success, but validation fails
 */
TEST_F(BaseHpccBenchmarkTest, SuccessfulExeFailedVal) {
    bm->returnValidate = false;
    EXPECT_FALSE(bm->executeBenchmark());
}

/**
 * Execute kernel fails
 */
TEST_F(BaseHpccBenchmarkTest, FailedExe) {
    bm->returnExecuteKernel = false;
    EXPECT_FALSE(bm->executeBenchmark());
}

/**
 * Benchmark Setup is successful with default data
 */
TEST(SetupTest, BenchmarkSetupIsSuccessful) {
    std::unique_ptr<MinimalBenchmark> bm = std::unique_ptr<MinimalBenchmark>(new MinimalBenchmark());
    EXPECT_TRUE(bm->setupBenchmark(global_argc, global_argv));
}

/**
 * Benchmark Setup fails because of failing configuration check
 */
TEST(SetupTest, BenchmarkConfigurationFailsSetup) {
    std::unique_ptr<MinimalBenchmark> bm = std::unique_ptr<MinimalBenchmark>(new MinimalBenchmark());
    bm->configurationCheckSucceeds = false;
    EXPECT_FALSE(bm->setupBenchmark(global_argc, global_argv));
}

/**
 * Benchmark Execution fails if configuration check failed
 */
TEST(SetupTest, BenchmarkConfigurationFailsExecution) {
    std::unique_ptr<MinimalBenchmark> bm = std::unique_ptr<MinimalBenchmark>(new MinimalBenchmark());
    bm->configurationCheckSucceeds = false;
    bm->setupBenchmark(global_argc, global_argv);
    EXPECT_FALSE(bm->executeBenchmark());
}

/**
 * Benchmark Setup fails with empty data
 */
TEST(SetupTest, BenchmarkSetupFails) {
    std::unique_ptr<MinimalBenchmark> bm = std::unique_ptr<MinimalBenchmark>(new MinimalBenchmark());
    char** tmp_argv = new char*[2];
    char* name_str = new char[5];
    strcpy(name_str, "name");
    tmp_argv[0] = name_str;
    tmp_argv[1] = nullptr;
    EXPECT_FALSE(bm->setupBenchmark(1, tmp_argv));
    delete [] tmp_argv;
    delete [] name_str;
}
