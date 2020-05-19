//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "linpack_benchmark.hpp"
#include <algorithm>
#ifdef _INTEL_MKL_
#include "mkl.h"
#endif

struct LinpackKernelSeparateTest : testing::Test, testing::WithParamInterface<std::string> {

    HOST_DATA_TYPE *A, *B, *C, *scale;
    cl_int *ipvt;
    uint array_size;
    std::unique_ptr<linpack::LinpackBenchmark> bm;

    LinpackKernelSeparateTest() {
        int argc = 3;
        std::string str_param = GetParam();
        std::vector<char> param(str_param.c_str(), str_param.c_str() + str_param.size() + 1);
        char* argv[3] = {"Test", "-f", reinterpret_cast<char*>(&param)};
        array_size = (1 << LOCAL_MEM_BLOCK_LOG);
        bm->getExecutionSettings().programSettings->matrixSize = array_size;
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(argc, argv));
        posix_memalign(reinterpret_cast<void **>(&A), 4096,
                       sizeof(HOST_DATA_TYPE) * array_size * array_size);
        posix_memalign(reinterpret_cast<void **>(&B), 4096,
                       sizeof(HOST_DATA_TYPE) * array_size * array_size);
        posix_memalign(reinterpret_cast<void **>(&C), 4096,
                       sizeof(HOST_DATA_TYPE) * array_size * array_size);
        posix_memalign(reinterpret_cast<void **>(&scale), 4096,
                       sizeof(HOST_DATA_TYPE) * array_size );
        posix_memalign(reinterpret_cast<void **>(&ipvt), 4096,
                       sizeof(cl_int) * array_size);  
    }

    void initializeData() {
        int init = 1325;
        for (int i=0; i<array_size; i++) {
            for (int j=0; j < array_size; j++) {
                init = 3125 * init % 65536;
                A[i* array_size + j] = (init - 32768.0) / 16384.0;
                init = 3125 * init % 65536;
                B[i * array_size + j] = (init - 32768.0) / 16384.0;
                C[i* array_size +j] = 1.0;
            }
            ipvt[i] = i;
        }
        for (int i=0; i<array_size; i++) {
            //Too small values here lead to big errors in FP operations especially for C2
            // This is most likely also the reason for poor accuracy, since only pairwise pivoting is used
            A[i * array_size + i] = 2.0;
            scale[i] = -1.0/A[i * array_size + i];
        }
    }

    void executeTest(std::string kernel_name) {
        int err;

        // Create Command queue
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device);

        // Create Buffers for input and output
        cl::Buffer Buffer_a(*bm->getExecutionSettings().context, CL_MEM_READ_WRITE,
                            sizeof(HOST_DATA_TYPE)*array_size*array_size);
        cl::Buffer Buffer_b(*bm->getExecutionSettings().context, CL_MEM_READ_WRITE,
                            sizeof(HOST_DATA_TYPE)*array_size*array_size);
        cl::Buffer Buffer_c(*bm->getExecutionSettings().context, CL_MEM_READ_WRITE,
                            sizeof(HOST_DATA_TYPE)*array_size*array_size);
        cl::Buffer Buffer_scale(*bm->getExecutionSettings().context, CL_MEM_READ_WRITE,
                            sizeof(HOST_DATA_TYPE)*array_size);
        cl::Buffer Buffer_pivot(*bm->getExecutionSettings().context, CL_MEM_READ_WRITE,
                                sizeof(cl_int)*array_size);

        // create the kernels
        cl::Kernel test_c4_kernel(*bm->getExecutionSettings().program, kernel_name.c_str(),
                                  &err);
        ASSERT_CL(err);


        // prepare kernels
        err = test_c4_kernel.setArg(0, Buffer_a);
        ASSERT_CL(err);
        err = test_c4_kernel.setArg(1, Buffer_b);
        ASSERT_CL(err);
        err = test_c4_kernel.setArg(2, Buffer_c);
        ASSERT_CL(err);
        err = test_c4_kernel.setArg(3, Buffer_scale);
        ASSERT_CL(err);
        err = test_c4_kernel.setArg(4, Buffer_pivot);
        ASSERT_CL(err);
        err = test_c4_kernel.setArg(5, static_cast<uint>(array_size >> LOCAL_MEM_BLOCK_LOG));
        ASSERT_CL(err);

        /* --- Execute actual benchmark kernels --- */

        double t;
        std::vector<double> executionTimes;
        for (int i = 0; i < bm->getExecutionSettings().programSettings->numRepetitions; i++) {
            compute_queue.enqueueWriteBuffer(Buffer_a, CL_TRUE, 0,
                                             sizeof(HOST_DATA_TYPE)*array_size*array_size, A);
            compute_queue.enqueueWriteBuffer(Buffer_b, CL_TRUE, 0,
                                             sizeof(HOST_DATA_TYPE)*array_size*array_size, B);
            compute_queue.enqueueWriteBuffer(Buffer_c, CL_TRUE, 0,
                                             sizeof(HOST_DATA_TYPE)*array_size*array_size, C);
            compute_queue.enqueueWriteBuffer(Buffer_scale, CL_TRUE, 0,
                                             sizeof(HOST_DATA_TYPE)*array_size, scale);
            compute_queue.enqueueWriteBuffer(Buffer_pivot, CL_TRUE, 0,
                                             sizeof(cl_int)*array_size, ipvt);
            compute_queue.finish();
            auto t1 = std::chrono::high_resolution_clock::now();
            compute_queue.enqueueTask(test_c4_kernel);
            compute_queue.finish();
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> timespan =
                    std::chrono::duration_cast<std::chrono::duration<double>>
                                                                           (t2 - t1);
            executionTimes.push_back(timespan.count());
        }

        /* --- Read back results from Device --- */
        compute_queue.enqueueReadBuffer(Buffer_a, CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*array_size*array_size, A);
        compute_queue.enqueueReadBuffer(Buffer_b, CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*array_size*array_size, B);
        compute_queue.enqueueReadBuffer(Buffer_c, CL_TRUE, 0,
                                        sizeof(HOST_DATA_TYPE)*array_size*array_size, C);
    }

    ~LinpackKernelSeparateTest() override {
        free(A);
        free(B);
        free(C);
        free(ipvt);
        free(scale);
    }
};

/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelSeparateTest, FPGACorrectResultsForC1) {
    auto a_result = new HOST_DATA_TYPE[array_size * array_size];
    initializeData();
    executeTest("test_c1");
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            a_result[i * array_size + j] = A[i * array_size + j];
        }
    }
    initializeData();
    linpack::gefa_ref(A, array_size, array_size, ipvt);
    double error = 0.0;
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            error += std::fabs(a_result[i * array_size + j] - A[i * array_size + j]);
        }
    }
    double normalized_error = error / (std::numeric_limits<HOST_DATA_TYPE>::epsilon() * array_size * array_size);
    EXPECT_LT(normalized_error, 5.0);
    delete [] a_result;
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelSeparateTest, FPGACorrectResultsForC2) {
    auto b_result = new HOST_DATA_TYPE[array_size * array_size];
    initializeData();
    executeTest("test_c2");
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            b_result[i * array_size + j] = B[i * array_size + j];
        }
    }
    initializeData();
    for (int k=0; k<array_size; k++) {
        for (int i=0; i < array_size; i++) {
            B[i * array_size + k] = -B[i * array_size + k] / A[k * array_size + k];
        }
        for (int j=k+1; j < array_size; j++) {
            for (int i=0; i < array_size; i++) {
                B[i * array_size + j] += B[i * array_size + k] * A[k * array_size + j];
            }
        }
    }
    double error = 0.0;
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            error += std::fabs(b_result[i * array_size + j] - B[i * array_size + j]);
        }
    }
    double normalized_error = error / (std::numeric_limits<HOST_DATA_TYPE>::epsilon() * array_size * array_size);
    EXPECT_LT(normalized_error, 5.0);
    delete [] b_result;
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelSeparateTest, FPGACorrectResultsForC3) {
    auto b_result = new HOST_DATA_TYPE[array_size * array_size];
    initializeData();
    executeTest("test_c3");
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            b_result[i * array_size + j] = B[i * array_size + j];
        }
    }
    initializeData();
    for (int k=0; k<array_size; k++) {
        for (int j=0; j < array_size; j++) {
            for (int i=k+1; i < array_size; i++) {
                B[i * array_size + j] += B[k * array_size + j] * A[i * array_size + k];
            }
        }
    }
    double error = 0.0;
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            error += std::fabs(b_result[i * array_size + j] - B[i * array_size + j]);
        }
    }
    double normalized_error = error / (std::numeric_limits<HOST_DATA_TYPE>::epsilon() * array_size * array_size);
    EXPECT_LT(normalized_error, 5.0);
    delete [] b_result;
}

/**
 * Execution returns correct results for a single repetition
 */
TEST_P(LinpackKernelSeparateTest, FPGACorrectResultsForC4) {
    auto c_result = new HOST_DATA_TYPE[array_size * array_size];
    initializeData();
    executeTest("test_c4");
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            c_result[i * array_size + j] = C[i * array_size + j];
        }
    }
    initializeData();
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            for (int k=0; k < array_size; k++) {
                C[i* array_size + j] += A[i * array_size + k] * B[k*array_size + j];
            }
        }
    }
    double error = 0.0;
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            error += std::fabs(c_result[i * array_size + j] - C[i * array_size + j]);
        }
    }
    std::cout << "Total error: " << error << std::endl;
    double normalized_error = error / (std::numeric_limits<HOST_DATA_TYPE>::epsilon() * array_size * array_size);
    EXPECT_LT(normalized_error, 10.0);
    EXPECT_LT(error, 1.0e-3);
    delete [] c_result;
}

#ifdef INTEL_FPGA
INSTANTIATE_TEST_CASE_P(Default, LinpackKernelSeparateTest,
        testing::Values("lu_blocked_pvt_test_emulate.aocx")
);
#endif


#ifdef XILINX_FPGA
// TODO: separate function testing is disabled since compilation from
// multiple cl files fails with the given code because of redundant
// kernel names
INSTANTIATE_TEST_CASE_P(DISABLED_Default, LinpackKernelSeparateTest,
        testing::Values("lu_blocked_pvt_test_emulate.xclbin")
);
#endif

