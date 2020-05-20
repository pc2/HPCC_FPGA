//
// Created by Marius Meyer on 04.12.19.
//
#include <memory>

#include "gtest/gtest.h"
#include "fft_benchmark.hpp"
#include "parameters.h"
#include "test_program_settings.h"


struct FFTKernelTest : testing::Test {
    std::unique_ptr<fft::FFTBenchmark> bm;
    std::unique_ptr<fft::FFTData> data;

    FFTKernelTest() {
        bm = std::unique_ptr<fft::FFTBenchmark>(new fft::FFTBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
        bm->getExecutionSettings().programSettings->inverse = false;
        data = bm->generateInputData();
    }

    ~FFTKernelTest() override {
        bm = nullptr;
        data = nullptr;
    }

};


/**
 * Tests if calculate returns the correct execution results
 */
TEST_F(FFTKernelTest, CalculateReturnsCorrectExecutionResultFor11False) {
    bm->getExecutionSettings().programSettings->numRepetitions = 1;
    data = bm->generateInputData();
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(1, result->timings.size());
}

/**
 * Tests if calculate returns the correct execution results for multiple repetitions
 */
TEST_F(FFTKernelTest, CalculateReturnsCorrectExecutionResultFor24True) {
    bm->getExecutionSettings().programSettings->numRepetitions = 2;
    data = bm->generateInputData();
    auto result = bm->executeKernel(*data);
    EXPECT_EQ(2, result->timings.size());
}

/**
 * Check if FFT of zeros returns just zeros
 */
TEST_F(FFTKernelTest, FFTReturnsZero) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(0.0);
        data->data[i].imag(0.0);
    }
    auto result = bm->executeKernel(*data);
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(std::abs(data->data[i]), 0.0);
    }
}


/**
 * Check if FFT calculates the correct result for all number being 1.0,1.0i
 */
TEST_F(FFTKernelTest, FFTCloseToZeroForAll1And1) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(1.0);
        data->data[i].imag(1.0);
    }
    auto result = bm->executeKernel(*data);
    EXPECT_NEAR(data->data[0].real(), (1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data->data[0].imag(), (1 << LOG_FFT_SIZE), 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data->data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data->data[i].imag(), 0.0, 0.00001);
    }
}


/**
* Check if iFFT calculates the correct result for all number being 1.0,1.0i
*/
TEST_F(FFTKernelTest, IFFTCloseToZeroForAll1And1) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(1.0);
        data->data[i].imag(0.0);
    }
    auto result = bm->executeKernel(*data);
    EXPECT_NEAR(data->data[0].real(), static_cast<HOST_DATA_TYPE>(1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data->data[0].imag(), 0.0, 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data->data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data->data[i].imag(), 0.0, 0.00001);
    }
}

/**
 * Check if calling FFt and iFFT result in data that is close to the original data with small error
 */
TEST_F(FFTKernelTest, FFTandiFFTProduceResultCloseToSource) {
    auto verify_data = bm->generateInputData();

    auto result = bm->executeKernel(*data);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i] /=  (1 << LOG_FFT_SIZE);
    }

    // Need to again bit reverse input for iFFT
    fft::bit_reverse(data->data, 1);
    bm->getExecutionSettings().programSettings->inverse = true;
    auto result2 = bm->executeKernel(*data);
    // Since data was already sorted by iFFT the bit reversal of the kernel has t be undone
    fft::bit_reverse(data->data, 1);

    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data->data[i]), std::abs(verify_data->data[i]), 0.001);
    }
}

/**
 * Check if FPGA FFT and reference FFT give the same results
 */
TEST_F(FFTKernelTest, FPGAFFTAndCPUFFTGiveSameResults) {
    auto verify_data = bm->generateInputData();

    auto result = bm->executeKernel(*data);

    fft::fourier_transform_gold(false,LOG_FFT_SIZE,verify_data->data);
    fft::bit_reverse(verify_data->data, 1);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i] -= verify_data->data[i];
    }
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data->data[i]), 0.0, 0.001);
    }
}

/**
 * Check if FPGA iFFT and reference iFFT give the same results
 */
TEST_F(FFTKernelTest, FPGAiFFTAndCPUiFFTGiveSameResults) {
    auto verify_data = bm->generateInputData();

    bm->getExecutionSettings().programSettings->inverse = true;
    auto result = bm->executeKernel(*data);

    fft::fourier_transform_gold(true,LOG_FFT_SIZE,verify_data->data);
    fft::bit_reverse(verify_data->data, 1);

    // Normalize iFFT result
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i] -= verify_data->data[i];
    }
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data->data[i]), 0.0, 0.001);
    }
}
