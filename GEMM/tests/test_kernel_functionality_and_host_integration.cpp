//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "../src/host/execution.h"
#include "../src/host/gemm_functionality.hpp"
#include "parameters.h"
#include "../src/host/setup/fpga_setup.hpp"

void
ref_matmul(HOST_DATA_TYPE const* A, HOST_DATA_TYPE const* B, HOST_DATA_TYPE* C, uint size) {
    for (int i=0; i< size; i++) {
        for (int j=0; j< size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}


struct OpenCLKernelTest : testing::Test {
    std::string kernelFileName;
    HOST_DATA_TYPE *A;
    HOST_DATA_TYPE *B;
    HOST_DATA_TYPE *C;
    HOST_DATA_TYPE *C_out;
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    cl_uint matrix_size;

    OpenCLKernelTest() {
        kernelFileName = "gemm_cannon_emulate.aocx";
        matrix_size = 2 * BLOCK_SIZE;
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&B), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&C), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&C_out), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        setupFPGA();
    }

    void setupFPGA() {
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        KERNEL_NAME,
                        1,
                        matrix_size,
                        false
                });
        HOST_DATA_TYPE norm;
        matgen(A,1,matrix_size, matrix_size, &norm);
        matgen(B,2,matrix_size, matrix_size, &norm);
        matgen(C,3,matrix_size, matrix_size, &norm);
    }

    ~OpenCLKernelTest() override {
        free(A);
        free(B);
        free(C);
        free(C_out);
    }
};

struct DifferentOpenCLKernelTest : OpenCLKernelTest, testing::WithParamInterface<std::tuple<std::string, unsigned>> {
    DifferentOpenCLKernelTest() {
        auto params = GetParam();
        kernelFileName = std::get<0>(params);
        matrix_size = std::get<1>(params) * BLOCK_SIZE;
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&B), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&C), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        posix_memalign(reinterpret_cast<void **>(&C_out), 64,
                       sizeof(HOST_DATA_TYPE) * matrix_size * matrix_size);
        setupFPGA();
    }
};


/**
 * Tests if C will be multiplied by beta
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectCtimesBeta) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i * matrix_size + j] = 0.0;
            B[i * matrix_size + j] = 0.0;
            C[i * matrix_size + j] = 1.0;
        }
    }
    auto result = bm_execution::calculate(config, A, B, C, C_out,0.0,2.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_FLOAT_EQ(C_out[i * matrix_size + j], 2.0 * C[i * matrix_size + j]);
        }
    }
}

/**
 * Tests if A will be multiplied by alpha
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectAtimesAlpha) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            B[i * matrix_size + j] = i == j ? 1.0 : 0.0;
            C[i * matrix_size + j] = 0.0;
        }
    }
    auto result = bm_execution::calculate(config, A, B, C, C_out,2.0,0.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_FLOAT_EQ(C_out[i * matrix_size + j], 2.0 * A[i * matrix_size + j]);
        }
    }
}

/**
 * Tests if B will be multiplied by alpha
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectBtimesAlpha) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i * matrix_size + j] = i == j ? 1.0 : 0.0;
            C[i * matrix_size + j] = 0.0;
        }
    }
    auto result = bm_execution::calculate(config, A, B, C, C_out,2.0,0.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_FLOAT_EQ(C_out[i * matrix_size + j], 2.0 * B[i * matrix_size + j]);
        }
    }
}

/**
 * Tests if A will be multiplied with B
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectAmulB) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            C[i * matrix_size + j] = 0.0;
            A[i * matrix_size + j] = j;
            B[i * matrix_size + j] = i;
        }
    }
    auto result = bm_execution::calculate(config, A, B, C, C_out,1.0,0.0);

    HOST_DATA_TYPE c_ref_out[matrix_size * matrix_size];
    ref_matmul(A,B,c_ref_out,matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(C_out[i * matrix_size + j], c_ref_out[i * matrix_size + j], 0.001);
        }
    }
}

/**
 * Tests if C will be added to A
 */
TEST_P(DifferentOpenCLKernelTest, FPGACorrectCplusA) {

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            B[i * matrix_size + j] = i == j ? 1.0 : 0.0;
        }
    }

    auto result = bm_execution::calculate(config, A, B, C, C_out,1.0, 1.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_FLOAT_EQ(C_out[i * matrix_size + j], A[i * matrix_size + j] + C[i * matrix_size + j]);
        }
    }
}


/**
 * Tests full multiply add
 */

TEST_P(DifferentOpenCLKernelTest, FPGACorrectbetaCplusalphaAB) {
    HOST_DATA_TYPE c_ref_out[matrix_size * matrix_size];
    auto result = bm_execution::calculate(config, A, B, C, C_out,0.5, 2.0);
    gemm_ref(A,B,C,c_ref_out,matrix_size,0.5,2.0);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            EXPECT_NEAR(C_out[i * matrix_size + j], c_ref_out[i * matrix_size + j], 0.001);
        }
    }
}


INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
        testing::Combine(testing::Values("gemm_cannon_emulate.aocx"), testing::Values(1,2)
                        ));
