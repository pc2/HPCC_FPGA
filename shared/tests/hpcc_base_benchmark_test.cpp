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
    addAdditionalParseOptions(cxxopts::Options &options) override {}

public:

    bool returnInputData = true;  
    bool returnExecuteKernel = true; 
    bool returnValidate = true;

    std::unique_ptr<int>
    generateInputData() override { return returnInputData ? std::unique_ptr<int>(new int) : std::unique_ptr<int>(nullptr);}

    std::unique_ptr<int>
    executeKernel(int &data) override { return returnExecuteKernel ? std::unique_ptr<int>(new int) : std::unique_ptr<int>(nullptr);}

    bool
    validateOutputAndPrintError(int &data) override { return returnValidate;}

    void
    collectAndPrintResults(const int &output) override {}

    MinimalBenchmark() : HpccFpgaBenchmark(0, { nullptr}) {}

};


class SuccessBenchmark : public hpcc_base::HpccFpgaBenchmark<hpcc_base::BaseSettings, int, int> {

protected:

    void
    addAdditionalParseOptions(cxxopts::Options &options) override {}

public:

    bool returnInputData = true;  
    bool returnExecuteKernel = true; 
    bool returnValidate = true;

    std::unique_ptr<int>
    generateInputData() override { 
        if (!returnInputData) {
            throw fpga_setup::FpgaSetupException("Test input data failed");
        }
        return std::unique_ptr<int>(new int);}

    std::unique_ptr<int>
    executeKernel(int &data) override { 
        if (!returnExecuteKernel) {
            throw fpga_setup::FpgaSetupException("Test execute kernel failed");
        }
        return std::unique_ptr<int>(new int);}

    bool
    validateOutputAndPrintError(int &data) override { return returnValidate;}

    bool 
    setupBenchmark(const int argc, char const* const* argv) {
        try {

            // Create deep copies of the input parameters to prevent modification from 
            // the cxxopts library
            int tmp_argc = argc;
            auto tmp_argv = new char*[argc + 1];
            for (int i =0; i < argc; i++) {
                int len = strlen(argv[i]) + 1;
                tmp_argv[i] = new char[len];
                strcpy(tmp_argv[i], argv[i]);
            }
            tmp_argv[argc] = nullptr;

            std::unique_ptr<hpcc_base::BaseSettings> programSettings = parseProgramParameters(tmp_argc, tmp_argv);

            // for (int i =0; i < argc; i++) {
            //     delete [] tmp_argv[i];
            // }
            // delete [] tmp_argv;

            auto usedDevice = fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                                                programSettings->defaultDevice);

            auto context = std::unique_ptr<cl::Context>(new cl::Context(*usedDevice));
            auto program = fpga_setup::fpgaSetup(context.get(), {*usedDevice},
                                                                &programSettings->kernelFileName);

            executionSettings = std::unique_ptr<hpcc_base::ExecutionSettings<hpcc_base::BaseSettings>>(new hpcc_base::ExecutionSettings<hpcc_base::BaseSettings>(std::move(programSettings), std::move(usedDevice), 
                                                                std::move(context), std::move(program)));
            // Get the rank of the process
            int world_rank = 0;

#ifdef _USE_MPI_
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
            if (world_rank == 0) {
                printFinalConfiguration(*executionSettings);
            }

            return true;
        }
        catch (fpga_setup::FpgaSetupException e) {
            std::cerr << "An error occured while setting up the benchmark: " << std::endl;
            std::cerr << "\t" << e.what() << std::endl;
            return false;
        }
    }

    void
    collectAndPrintResults(const int &output) override {}

    SuccessBenchmark() : HpccFpgaBenchmark(0, { nullptr}) {}

};

class BaseHpccBenchmarkTest :public  ::testing::Test {

public:
    std::unique_ptr<SuccessBenchmark> bm;

    BaseHpccBenchmarkTest() {
        bm = std::unique_ptr<SuccessBenchmark>(new SuccessBenchmark());
        bool success = bm->setupBenchmark(global_argc, global_argv);
        EXPECT_TRUE(success);
    }

};

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
