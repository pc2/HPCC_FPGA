//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "../src/host/execution.h"
#include "setup/fpga_setup.hpp"
#include "../src/host/random_access_functionality.hpp"


struct OpenCLKernelTest : testing::Test {
    HOST_DATA_TYPE *data;
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    cl_uint array_size;

    OpenCLKernelTest() {
        array_size =  128 * NUM_KERNEL_REPLICATIONS * BUFFER_SIZE;
        posix_memalign(reinterpret_cast<void **>(&data), 4096,
                       sizeof(HOST_DATA_TYPE) * array_size);
    }

    void setupFPGA(std::string kernelFileName) {
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        1,
                        NUM_KERNEL_REPLICATIONS,
                        array_size
                });
        generateInputData(data, array_size);
    }

    ~OpenCLKernelTest() override {
        free(data);
    }
};

struct DifferentOpenCLKernelTest : OpenCLKernelTest, testing::WithParamInterface<std::string> {
    DifferentOpenCLKernelTest() {
        auto params = GetParam();
        auto kernel_file = params;
        setupFPGA(kernel_file);
    }
};

/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectNumberOfMeasurements1Rep) {

    auto result = bm_execution::calculate(config, data);
    EXPECT_EQ(result->times.size(), 1);
}

/**
 * Check if the number of measurements from the calculation matches the number of repetitions
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectNumberOfMeasurements3Rep) {
    config->repetitions = 3;
    auto result = bm_execution::calculate(config, data);
    EXPECT_EQ(result->times.size(), 3);
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_P(DifferentOpenCLKernelTest, FPGAErrorBelow1Percent) {

    auto result = bm_execution::calculate(config, data);
    double errors = checkRandomAccessResults(data, array_size);
    EXPECT_LT(errors, 0.01);
}


#ifdef INTEL_FPGA
INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
        testing::Values("random_access_kernels_single_emulate.aocx")
);
#endif

#ifdef XILINX_FPGA
INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
                        testing::Values("random_access_kernels_single_emulate.xclbin")
);
#endif