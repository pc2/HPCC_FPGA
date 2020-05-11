//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "../src/host/execution.h"
#include "setup/fpga_setup.hpp"
#include "testing/test_program_settings.h"
#include "../src/host/stream_functionality.hpp"


struct OpenCLKernelTest :public  ::testing::Test {
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

    void SetUp( ) { 
        std::cout << programSettings << std::endl;
       setupFPGA(programSettings);
   }

    void setupFPGA(std::shared_ptr<ProgramSettings> settings) {
        // Redirect stout buffer to local buffer to make checks possible
        std::stringstream newStdOutBuffer;
        std::streambuf *oldStdOutBuffer = std::cout.rdbuf();
        std::cout.rdbuf(newStdOutBuffer.rdbuf());

        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(settings->defaultPlatform, settings->defaultDevice);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &settings->kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        1,
                        NUM_KERNEL_REPLICATIONS,
                        array_size,
                        false,
                        settings->useSingleKernel
                });
        HOST_DATA_TYPE norm;
        generateInputData(A, B, C, array_size);

        // Redirect stdout to old buffer
        std::cout.rdbuf(oldStdOutBuffer);
    }

    ~OpenCLKernelTest() override {
        free(A);
        free(B);
        free(C);
    }
};


/**
 * Execution returns correct results for a single repetition
 */
TEST_F(OpenCLKernelTest, FPGACorrectResultsOneRepetition) {

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
TEST_F(OpenCLKernelTest, FPGACorrectResultsThreeRepetition) {
    config->repetitions = 3;
    auto result = bm_execution::calculate(config, A, B, C);
    for (int i = 0; i < array_size; i++) {
        EXPECT_FLOAT_EQ(A[i], 6750.0);
        EXPECT_FLOAT_EQ(B[i], 1350.0);
        EXPECT_FLOAT_EQ(C[i], 1800.0);
    }
}
