//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "../src/host/execution.h"
#include "../src/host/setup/fpga_setup.hpp"
#include "../src/host/stream_functionality.hpp"


struct OpenCLKernelTest : testing::Test {
    HOST_DATA_TYPE *A;
    HOST_DATA_TYPE *B;
    HOST_DATA_TYPE *C;
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    cl_uint array_size;

    OpenCLKernelTest() {
        array_size = VECTOR_COUNT * UNROLL_COUNT * NUM_KERNEL_REPLICATIONS * BUFFER_SIZE;
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                       sizeof(HOST_DATA_TYPE) * array_size);
        posix_memalign(reinterpret_cast<void **>(&B), 64,
                       sizeof(HOST_DATA_TYPE) * array_size);
        posix_memalign(reinterpret_cast<void **>(&C), 64,
                       sizeof(HOST_DATA_TYPE) * array_size);
    }

    void setupFPGA(std::string kernelFileName, bool is_single_kernel) {
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        1,
                        NUM_KERNEL_REPLICATIONS,
                        array_size,
                        false,
                        is_single_kernel
                });
        HOST_DATA_TYPE norm;
        generateInputData(A, B, C, array_size);
    }

    ~OpenCLKernelTest() override {
        free(A);
        free(B);
        free(C);
    }
};

struct DifferentOpenCLKernelTest : OpenCLKernelTest, testing::WithParamInterface<std::tuple<std::string,bool>> {
    DifferentOpenCLKernelTest() {
        auto params = GetParam();
        auto kernel_file = std::get<0>(params);
        bool is_single_kernel = std::get<1>(params);
        setupFPGA(kernel_file, is_single_kernel);
    }
};


/**
 * Execution returns correct results for a single repetition
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectResultsOneRepetition) {

    auto result = bm_execution::calculate(config, A, B, C);
    for (int i = 0; i < array_size; i++) {
        EXPECT_FLOAT_EQ(A[i], 30.0);
        EXPECT_FLOAT_EQ(B[i], 6.0);
        EXPECT_FLOAT_EQ(C[i], 8.0);
    }
}

/**
 * Execution returns correct results for three repetitions
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectResultsThreeRepetition) {
    config->repetitions = 3;
    auto result = bm_execution::calculate(config, A, B, C);
    for (int i = 0; i < array_size; i++) {
        EXPECT_FLOAT_EQ(A[i], 6750.0);
        EXPECT_FLOAT_EQ(B[i], 1350.0);
        EXPECT_FLOAT_EQ(C[i], 1800.0);
    }
}


#ifdef INTEL_FPGA
INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
        testing::Values(std::make_tuple("stream_kernels_emulate.aocx", false),
                std::make_tuple("stream_kernels_single_emulate.aocx", true))
);
#endif

#ifdef XILINX_FPGA
INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
                        testing::Values(std::make_tuple("stream_kernels_single_emulate.xclbin", true))
);
#endif