//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "../src/host/execution.h"
#include "parameters.h"
#include "../src/host/setup/fpga_setup.hpp"
#include "../src/host/fft_functionality.hpp"


struct OpenCLKernelTest : testing::Test {
    std::string kernelFileName = "fft1d_float_8_emulate.aocx";
    std::shared_ptr<bm_execution::ExecutionConfiguration> config;
    unsigned repetitions = 10;

    void setupFPGA() {
        std::vector<cl::Device> device = fpga_setup::selectFPGADevice(DEFAULT_PLATFORM, DEFAULT_DEVICE);
        cl::Context context(device[0]);
        cl::Program program = fpga_setup::fpgaSetup(&context, device, &kernelFileName);
        config = std::make_shared<bm_execution::ExecutionConfiguration>(
                bm_execution::ExecutionConfiguration{
                        context, device[0], program,
                        repetitions
                });
    }
};

/**
 * Parametrized test takes a tuple of 1 parameter:
 * - name of the emulation bitstream
 */
struct DifferentOpenCLKernelTest : OpenCLKernelTest, testing::WithParamInterface<std::string> {
    DifferentOpenCLKernelTest() {
        auto params = GetParam();
        kernelFileName = params;
        setupFPGA();
    }
};


/**
 * Tests if calculate returns the correct execution results
 */
TEST_P(DifferentOpenCLKernelTest, CalculateReturnsCorrectExecutionResultFor11False) {
    config->repetitions = 1;
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    auto result = bm_execution::calculate(config, data, 1, false);
    EXPECT_EQ(1, result->iterations);
    EXPECT_EQ(false, result->inverse);
    EXPECT_EQ(1, result->calculationTimings.size());
}

/**
 * Tests if calculate returns the correct execution results for multiple repetitions
 */
TEST_P(DifferentOpenCLKernelTest, CalculateReturnsCorrectExecutionResultFor24True) {
    config->repetitions = 2;
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE) * 4);
    auto result = bm_execution::calculate(config, data, 4, true);
    EXPECT_EQ(4, result->iterations);
    EXPECT_EQ(true, result->inverse);
    EXPECT_EQ(2, result->calculationTimings.size());
}

/**
 * Check if FFT of zeros returns just zeros
 */
TEST_P (DifferentOpenCLKernelTest, FFTReturnsZero) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(0.0);
        data[i].imag(0.0);
    }
    auto result = bm_execution::calculate(config, data, 1, false);
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(std::abs(data[i]), 0.0);
    }
    free(data);
}


/**
 * Check if FFT calculates the correct result for all number being 1.0,1.0i
 */
TEST_P (DifferentOpenCLKernelTest, FFTCloseToZeroForAll1And1) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(1.0);
        data[i].imag(1.0);
    }
    auto result = bm_execution::calculate(config, data, 1, false);
    EXPECT_NEAR(data[0].real(), (1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data[0].imag(), (1 << LOG_FFT_SIZE), 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data[i].imag(), 0.0, 0.00001);
    }
    free(data);
}


/**
* Check if iFFT calculates the correct result for all number being 1.0,1.0i
*/
TEST_P (DifferentOpenCLKernelTest, IFFTCloseToZeroForAll1And1) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(1.0);
        data[i].imag(0.0);
    }
    auto result = bm_execution::calculate(config, data, 1, true);
    EXPECT_NEAR(data[0].real(), static_cast<HOST_DATA_TYPE>(1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data[0].imag(), 0.0, 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data[i].imag(), 0.0, 0.00001);
    }
    free(data);
}

/**
 * Check if calling FFt and iFFT result in data that is close to the original data with small error
 */
TEST_P (DifferentOpenCLKernelTest, FFTandiFFTProduceResultCloseToSource) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    std::complex<HOST_DATA_TYPE> * verify_data;
    posix_memalign(reinterpret_cast<void**>(&verify_data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));

    generateInputData(data, 1);
    generateInputData(verify_data, 1);

    auto result = bm_execution::calculate(config, data, 1, false);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i] /=  (1 << LOG_FFT_SIZE);
    }

    // Need to again bit reverse input for iFFT
    bit_reverse(data, 1);
    auto result2 = bm_execution::calculate(config, data, 1, true);
    // Since data was already sorted by iFFT the bit reversal of the kernel has t be undone
    bit_reverse(data, 1);

    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data[i]), std::abs(verify_data[i]), 0.001);
    }
    free(data);
    free(verify_data);
}

/**
 * Check the included FFT error function on the host code on data produced by FFT
 */
TEST_P (DifferentOpenCLKernelTest, FFTErrorCheck) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    std::complex<HOST_DATA_TYPE> * verify_data;
    posix_memalign(reinterpret_cast<void**>(&verify_data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));

    generateInputData(data, 1);
    generateInputData(verify_data, 1);

    auto result = bm_execution::calculate(config, data, 1, false);

    // Need to again bit reverse input for iFFT
    double error = checkFFTResult(verify_data, data, 1);

    EXPECT_NEAR(error, 0.0, 1.0);

    free(data);
    free(verify_data);
}

/**
 * Check if FPGA FFT and reference FFT give the same results
 */
TEST_P (DifferentOpenCLKernelTest, FPGAFFTAndCPUFFTGiveSameResults) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    std::complex<HOST_DATA_TYPE> * data2;
    posix_memalign(reinterpret_cast<void**>(&data2), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));

    generateInputData(data, 1);
    generateInputData(data2, 1);

    auto result = bm_execution::calculate(config, data, 1, false);

    fourier_transform_gold(false,LOG_FFT_SIZE,data2);
    bit_reverse(data2, 1);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i] -= data2[i];
    }
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data[i]), 0.0, 0.001);
    }
    free(data);
    free(data2);
}

/**
 * Check if FPGA iFFT and reference iFFT give the same results
 */
TEST_P (DifferentOpenCLKernelTest, FPGAiFFTAndCPUiFFTGiveSameResults) {
    std::complex<HOST_DATA_TYPE> * data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));
    std::complex<HOST_DATA_TYPE> * data2;
    posix_memalign(reinterpret_cast<void**>(&data2), 64, sizeof(std::complex<HOST_DATA_TYPE>) * (1 << LOG_FFT_SIZE));

    generateInputData(data, 1);
    generateInputData(data2, 1);

    auto result = bm_execution::calculate(config, data, 1, true);

    fourier_transform_gold(true,LOG_FFT_SIZE,data2);
    bit_reverse(data2, 1);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i] -= data2[i];
    }
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data[i]), 0.0, 0.001);
    }
    free(data);
    free(data2);
}

INSTANTIATE_TEST_CASE_P(Default, DifferentOpenCLKernelTest,
                        testing::Values("fft1d_float_8_emulate.aocx"));
