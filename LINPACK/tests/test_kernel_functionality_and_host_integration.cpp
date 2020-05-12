//
// Created by Marius Meyer on 14.02.20.
//
#include "gtest/gtest.h"
#include "parameters.h"
#include "../src/host/execution.h"
#include "setup/fpga_setup.hpp"
#include "testing/test_program_settings.h"
#include "../src/host/linpack_functionality.hpp"
#ifdef _INTEL_MKL_
#include "mkl.h"
#endif
struct OpenCLKernelTest : testing::Test {
    HOST_DATA_TYPE *A;
    HOST_DATA_TYPE *b;
    cl_int *ipvt;
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    cl_uint array_size;
    std::string lastKernelFileName;

    OpenCLKernelTest() {
        array_size = (1 << LOCAL_MEM_BLOCK_LOG);
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                       sizeof(HOST_DATA_TYPE) * array_size * array_size);
        posix_memalign(reinterpret_cast<void **>(&b), 64,
                       sizeof(HOST_DATA_TYPE) * array_size);
        posix_memalign(reinterpret_cast<void **>(&ipvt), 64,
                       sizeof(cl_int) * array_size);
        setupFPGA(programSettings->kernelFileName);
    }

    void setupFPGA(std::string kernelFileName) {
        lastKernelFileName = kernelFileName;
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(programSettings->defaultPlatform, programSettings->defaultDevice);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        1,
                        array_size,
                });
        HOST_DATA_TYPE norm;
        generateInputData(A, b, ipvt, array_size, &norm);
    }

    ~OpenCLKernelTest() override {
        free(A);
        free(b);
        free(ipvt);
    }
};


/**
 * Execution returns correct results for a single repetition
 */
TEST_F(OpenCLKernelTest, FPGACorrectResultsOneRepetition) {

    auto result = bm_execution::calculate(config, A, b, ipvt);
    for (int i = 0; i < array_size; i++) {
        EXPECT_NEAR(b[i], 1.0, 1.0e-3);
    }
}

#ifdef __INTEL_MKL__
/**
 * Execution returns correct results for a single repetition
 */
TEST_F(OpenCLKernelTest, FPGASimilarResultsToLAPACKforSingleBlock) {

    auto result = bm_execution::calculate(config, A, b, ipvt);
    int info;    
    HOST_DATA_TYPE* bcpu = new HOST_DATA_TYPE[array_size];
    HOST_DATA_TYPE* Acpu = new HOST_DATA_TYPE[array_size * array_size];
    HOST_DATA_TYPE norm;
    generateInputData(A, bcpu, ipvt, array_size, &norm);
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            Acpu[i * array_size + j] = A[j* array_size + i];
        }
    }
    int s = static_cast<int>(array_size);
    int lrhs = 1;
#ifndef _DP
        sgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#else
        dgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#endif
    double error_emu = checkLINPACKresults(b, array_size);
    double error_cpu = checkLINPACKresults(bcpu, array_size);
    EXPECT_LE(error_emu, error_cpu+ 1.0);
    delete [] Acpu;
    delete [] bcpu;
}

/**
 * Execution of reference implementation returns correct results for a single repetition
 */
TEST_F(OpenCLKernelTest, FPGAReferenceImplSimilarToMKL) {

    gefa_ref(A, config->matrixSize, config->matrixSize, ipvt);
    gesl_ref(A, b, ipvt, config->matrixSize, config->matrixSize);
    int info;
    HOST_DATA_TYPE* bcpu = new HOST_DATA_TYPE[array_size];
    HOST_DATA_TYPE* Acpu = new HOST_DATA_TYPE[array_size * array_size];
    HOST_DATA_TYPE norm;
    generateInputData(A, bcpu, ipvt, array_size, &norm);
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            Acpu[i * array_size + j] = A[j* array_size + i];
        }
    }
    int s = static_cast<int>(array_size);
    int lrhs = 1;
#ifndef _DP
        sgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#else
        dgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#endif
    double error_emu = checkLINPACKresults(b, array_size);
    double error_cpu = checkLINPACKresults(bcpu, array_size);
    EXPECT_LE(error_emu, error_cpu+ 1.0);
    delete [] Acpu;
    delete [] bcpu;
}


/**
 * Execution returns correct results for a single repetition
 */
// TODO this test fails most likely because of inreasing errors in C2. Use partial pivoting or other mechanisms
//      to make the calculation stable again!
//      Remove DISABLED_ from test name to enable the test again.
TEST_F(OpenCLKernelTest, DISABLED_FPGASimilarResultsToLAPACKforMultipleBlocks) {
    free(A);
    free(b);
    free(ipvt);
    array_size = 4 * (1 << LOCAL_MEM_BLOCK_LOG);
    posix_memalign(reinterpret_cast<void **>(&A), 64,
                       sizeof(HOST_DATA_TYPE) * array_size * array_size);
    posix_memalign(reinterpret_cast<void **>(&b), 64,
                       sizeof(HOST_DATA_TYPE) * array_size);
    posix_memalign(reinterpret_cast<void **>(&ipvt), 64,
                       sizeof(cl_int) * array_size);
    setupFPGA(lastKernelFileName);
    auto result = bm_execution::calculate(config, A, b, ipvt);

    int info;
    HOST_DATA_TYPE* bcpu = new HOST_DATA_TYPE[array_size];
    HOST_DATA_TYPE* Acpu = new HOST_DATA_TYPE[array_size * array_size];
    HOST_DATA_TYPE norm;
    generateInputData(A, bcpu, ipvt, array_size, &norm);
    for (int i=0; i<array_size; i++) {
        for (int j=0; j < array_size; j++) {
            Acpu[i * array_size + j] = A[j* array_size + i];
        }
    }
    int s = static_cast<int>(array_size);
    int lrhs = 1;
#ifndef _DP
        sgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#else
        dgesv(&s, &lrhs, Acpu, &s, ipvt, bcpu, &s, &info);
#endif
    double error_emu = checkLINPACKresults(b, array_size);
    double error_cpu = checkLINPACKresults(bcpu, array_size);

    EXPECT_LE(error_emu, error_cpu + 1.0);

    delete [] Acpu;
    delete [] bcpu;
}


#endif
