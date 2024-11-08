//
// Created by Marius Meyer on 04.12.19
//


#include "gtest/gtest.h"
#include "setup/fpga_setup.hpp"
#include "test_program_settings.h"
#include "gmock/gmock.h"
#include "hpcc_benchmark.hpp"
#include "nlohmann/json.hpp"


// Dirty GoogleTest and static library hack
// Use this function in your test main
// This will make the linker link this library 
// and enable the included tests
void use_hpcc_base_lib() {}

template<class T>
class MinimalBenchmark : public hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, typename std::tuple_element<0, T>::type, typename std::tuple_element<1, T>::type, typename std::tuple_element<2, T>::type, int> {

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

    void
    executeKernel(int &data) override { return;}

    bool
    validateOutput(int &data) override { return returnValidate;}
    
    void
    printError() override {}

    bool
    checkInputParameters() override { return configurationCheckSucceeds;}

    void
    collectResults() override {}

    void
    printResults() override {}

    MinimalBenchmark() : hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, typename std::tuple_element<0, T>::type, typename std::tuple_element<1, T>::type, typename std::tuple_element<2, T>::type, int>(0, { nullptr}) {}

};

template<class TDevice, class TContext, class TProgram>
class SuccessBenchmark : public hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, TDevice, TContext, TProgram, int> {

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

    void
    executeKernel(int &data) override { 
        if (!returnExecuteKernel) {
            throw fpga_setup::FpgaSetupException("Test execute kernel failed");
        }
        executeKernelcalled++;
        return;}

    bool
    validateOutput(int &data) override { 
        validateOutputcalled++;
        return returnValidate;}
    
    void
    printError() override {}

    void
    collectResults() override {}

    void
    printResults() override {}

    bool
    checkInputParameters() override {
        if (forceSetupFail) {
            return false;
        }
        else {
            return hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, TDevice, TContext, TProgram, int>::checkInputParameters();
        }
    }

    SuccessBenchmark() : hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, TDevice, TContext, TProgram, int>(0, { nullptr}) {}

};

template<class T>
class BaseHpccBenchmarkTest :public  ::testing::Test {

using TDevice = typename std::tuple_element<0,T>::type;
using TContext = typename std::tuple_element<1,T>::type;
using TProgram = typename std::tuple_element<2,T>::type;

public:
    std::unique_ptr<SuccessBenchmark<TDevice, TContext, TProgram>> bm;

    BaseHpccBenchmarkTest() {
        bm = std::unique_ptr<SuccessBenchmark<TDevice, TContext, TProgram>>(new SuccessBenchmark<TDevice, TContext, TProgram>());
        bm->setupBenchmark(global_argc, global_argv);
    }

};

template<class T>
class SetupTest : public ::testing::Test {};

#ifdef USE_OCL_HOST
typedef ::testing::Types<std::tuple<cl::Device, cl::Context, cl::Program>> cl_types;
TYPED_TEST_SUITE(
        BaseHpccBenchmarkTest,
        cl_types);
TYPED_TEST_SUITE(
        SetupTest,
        cl_types);
#endif
#ifdef USE_XRT_HOST
#ifndef USE_ACCL
typedef ::testing::Types<std::tuple<xrt::device, bool, xrt::uuid>> xrt_types;
TYPED_TEST_SUITE(
        BaseHpccBenchmarkTest,
        xrt_types);
TYPED_TEST_SUITE(
        SetupTest,
        xrt_types);
#else
typedef ::testing::Types<std::tuple<xrt::device, fpga_setup::ACCLContext, xrt::uuid>> accl_types;
TYPED_TEST_SUITE(
        BaseHpccBenchmarkTest,
        accl_types);
TYPED_TEST_SUITE(
        SetupTest,
        accl_types);
#endif
#endif


TYPED_TEST(BaseHpccBenchmarkTest, SetupSucceedsForBenchmarkTest) {
        bool success = this->bm->setupBenchmark(global_argc, global_argv);
        EXPECT_TRUE(success);
}


/**
 * Checks if the testing flag works as expected
 */
TYPED_TEST(BaseHpccBenchmarkTest, AllExecutedWhenNotTestOnly) {
    this->bm->getExecutionSettings().programSettings->testOnly = false;
    this->bm->executeBenchmark();
    EXPECT_EQ(this->bm->validateOutputcalled, 1);
    EXPECT_EQ(this->bm->executeKernelcalled, 1);
    EXPECT_EQ(this->bm->generateInputDatacalled, 1);
}

TYPED_TEST(BaseHpccBenchmarkTest, NothingExecutedWhenTestOnly) {
    this->bm->getExecutionSettings().programSettings->testOnly = true;
    this->bm->executeBenchmark();
    EXPECT_EQ(this->bm->validateOutputcalled, 0);
    EXPECT_EQ(this->bm->executeKernelcalled, 0);
    EXPECT_EQ(this->bm->generateInputDatacalled, 0);
}

TYPED_TEST(BaseHpccBenchmarkTest, ExecutionSuccessWhenNotTestOnly) {
    this->bm->getExecutionSettings().programSettings->testOnly = false;
    EXPECT_TRUE(this->bm->executeBenchmark());

}

TYPED_TEST(BaseHpccBenchmarkTest, ExecutionFailsWhenTestOnlyAndSetupFails) {
    this->bm->getExecutionSettings().programSettings->testOnly = true;
    this->bm->forceSetupFail = true;
    this->bm->setupBenchmark(global_argc, global_argv);
    EXPECT_FALSE(this->bm->executeBenchmark());
}

TYPED_TEST(BaseHpccBenchmarkTest, ExecutionSuccessWhenTestOnlyAndSetupSuccess) {
    this->bm->getExecutionSettings().programSettings->testOnly = true;
    EXPECT_TRUE(this->bm->executeBenchmark());
}

/**
 * Checks if non existing device leads to an error
 */
TYPED_TEST(BaseHpccBenchmarkTest, FindNonExistingDevice) {
#ifdef USE_OCL_HOST
    ASSERT_THROW(fpga_setup::selectFPGADevice(this->bm->getExecutionSettings().programSettings->defaultPlatform, 100, this->bm->getExecutionSettings().programSettings->platformString).get(), fpga_setup::FpgaSetupException);
#else
    ASSERT_THROW(fpga_setup::selectFPGADevice(100).get(), fpga_setup::FpgaSetupException);
#endif
}

/**
 * Checks if using default platform and device is successful
 */
TYPED_TEST(BaseHpccBenchmarkTest, SuccessUseDefaultPlatformandDevice) {
#ifdef USE_OCL_HOST
    EXPECT_NE(fpga_setup::selectFPGADevice(this->bm->getExecutionSettings().programSettings->defaultPlatform, this->bm->getExecutionSettings().programSettings->defaultDevice, this->bm->getExecutionSettings().programSettings->platformString).get(), nullptr);
#else
    EXPECT_NE(fpga_setup::selectFPGADevice(this->bm->getExecutionSettings().programSettings->defaultDevice).get(), nullptr);
#endif
}

#ifdef USE_OCL_HOST
/**
 * Checks if non existing platform leads to an error
 */
TYPED_TEST(BaseHpccBenchmarkTest, FindNonExistingPlatform) {
    ASSERT_THROW(fpga_setup::selectFPGADevice(100, this->bm->getExecutionSettings().programSettings->defaultDevice, this->bm->getExecutionSettings().programSettings->platformString).get(), fpga_setup::FpgaSetupException);
}

/*
 * Check if wrong platform string leads to an error
 */
TYPED_TEST(BaseHpccBenchmarkTest, FindNonExistingPlatformString) {
    ASSERT_THROW(fpga_setup::selectFPGADevice(this->bm->getExecutionSettings().programSettings->defaultPlatform, this->bm->getExecutionSettings().programSettings->defaultDevice, "This is not a platform").get(), fpga_setup::FpgaSetupException);
}

#endif

/**
 * Execute kernel and validation is success
 */
TYPED_TEST(BaseHpccBenchmarkTest, SuccessfulExeAndVal) {
    EXPECT_TRUE(this->bm->executeBenchmark());
}

/**
 * Execute kernel is success, but validation fails
 */
TYPED_TEST(BaseHpccBenchmarkTest, SuccessfulExeFailedVal) {
    this->bm->returnValidate = false;
    EXPECT_FALSE(this->bm->executeBenchmark());
}

/**
 * Execute kernel fails
 */
TYPED_TEST(BaseHpccBenchmarkTest, FailedExe) {
    this->bm->returnExecuteKernel = false;
    EXPECT_FALSE(this->bm->executeBenchmark());
}

/**
 * Benchmark Setup is successful with default data
 */
TYPED_TEST(SetupTest, BenchmarkSetupIsSuccessful) {
    std::unique_ptr<MinimalBenchmark<TypeParam>> bm = std::unique_ptr<MinimalBenchmark<TypeParam>>(new MinimalBenchmark<TypeParam>());
    EXPECT_TRUE(bm->setupBenchmark(global_argc, global_argv));
}

/**
 * Benchmark Setup fails because of failing configuration check
 */
TYPED_TEST(SetupTest, BenchmarkConfigurationFailsSetup) {
    std::unique_ptr<MinimalBenchmark<TypeParam>> bm = std::unique_ptr<MinimalBenchmark<TypeParam>>(new MinimalBenchmark<TypeParam>());
    bm->configurationCheckSucceeds = false;
    EXPECT_FALSE(bm->setupBenchmark(global_argc, global_argv));
}

/**
 * Benchmark Execution fails if configuration check failed
 */
TYPED_TEST(SetupTest, BenchmarkConfigurationFailsExecution) {
    std::unique_ptr<MinimalBenchmark<TypeParam>> bm = std::unique_ptr<MinimalBenchmark<TypeParam>>(new MinimalBenchmark<TypeParam>());
    bm->configurationCheckSucceeds = false;
    bm->setupBenchmark(global_argc, global_argv);
    EXPECT_FALSE(bm->executeBenchmark());
}

/**
 * Benchmark Setup fails with empty data
 */
TYPED_TEST(SetupTest, BenchmarkSetupFails) {
    std::unique_ptr<MinimalBenchmark<TypeParam>> bm = std::unique_ptr<MinimalBenchmark<TypeParam>>(new MinimalBenchmark<TypeParam>());
    char** tmp_argv = new char*[2];
    char* name_str = new char[5];
    strcpy(name_str, "name");
    tmp_argv[0] = name_str;
    tmp_argv[1] = nullptr;
    EXPECT_FALSE(bm->setupBenchmark(1, tmp_argv));
    delete [] tmp_argv;
    delete [] name_str;
}

using json = nlohmann::json;

/**
 *
 * Check if dump-json flag produces valid json output
 */
TYPED_TEST(SetupTest, BenchmarkJsonDump) {
    std::unique_ptr<MinimalBenchmark<TypeParam>> bm = std::unique_ptr<MinimalBenchmark<TypeParam>>(new MinimalBenchmark<TypeParam>());
    bm->setupBenchmark(global_argc, global_argv);
    bm->getExecutionSettings().programSettings->dumpfilePath = "out.json";
    bm->executeBenchmark();
    std::FILE *f = std::fopen("out.json", "r");
    EXPECT_NE(f, nullptr);
    if (f != nullptr) {
        // json::parse will panic if f is nullptr
        json j = json::parse(f);
        // check if the expected keys are there
        EXPECT_TRUE(j.contains("config_time"));
        EXPECT_TRUE(j.contains("device"));
        EXPECT_TRUE(j.contains("environment"));
        EXPECT_TRUE(j.contains("git_commit"));
        EXPECT_TRUE(j.contains("results"));
        EXPECT_TRUE(j.contains("settings"));
        EXPECT_TRUE(j.contains("timings"));
        EXPECT_TRUE(j.contains("version"));
    }
}

