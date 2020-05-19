//
// Created by Marius Meyer on 20.01.20.
//

#include "gtest/gtest.h"
#include "fft_benchmark.hpp"
#include "parameters.h"
#include "test_program_settings.h"


struct FFTHostTest : testing::Test {
    std::unique_ptr<fft::FFTBenchmark> bm;
    std::unique_ptr<fft::FFTData> data;

    FFTHostTest() {
        bm = std::unique_ptr<fft::FFTBenchmark>(new fft::FFTBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->numRepetitions = 1;
        bm->getExecutionSettings().programSettings->inverse = false;
        data = bm->generateInputData();
    }

    ~FFTHostTest() override {
        bm = nullptr;
        data = nullptr;
    }

};

/**
 * Check if data generator generates reproducable inputs
 */
TEST_F(FFTHostTest, DataInputReproducible) {
    auto data2 = bm->generateInputData();
    for (int i=0; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(data->data[i].real(), data2->data[i].real());
        EXPECT_FLOAT_EQ(data->data[i].imag(), data2->data[i].imag());
    }
}

/**
 * Check if FFT of zeros returns just zeros
 */
TEST_F(FFTHostTest, FFTReturnsZero) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(0.0);
        data->data[i].imag(0.0);
    }
    fft::fourier_transform_gold(false, LOG_FFT_SIZE, data->data);
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(std::abs(data->data[i]), 0.0);
    }
}


/**
 * Check if FFT calculates the correct result for all number being 1.0,1.0i
 */
TEST_F(FFTHostTest, FFTCloseToZeroForAll1And1) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(1.0);
        data->data[i].imag(1.0);
    }
    fft::fourier_transform_gold(false, LOG_FFT_SIZE, data->data);
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
TEST_F(FFTHostTest, IFFTCloseToZeroForAll1And1) {
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data->data[i].real(1.0);
        data->data[i].imag(1.0);
    }
    fft::fourier_transform_gold(true, LOG_FFT_SIZE, data->data);
    EXPECT_NEAR(data->data[0].real(), (1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data->data[0].imag(), (1 << LOG_FFT_SIZE), 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data->data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data->data[i].imag(), 0.0, 0.00001);
    }
}


/**
 * Check if FFT and FFT check give low error when FFT is calculated directly
 */
TEST_F(FFTHostTest, FFTandiFFTProduceResultCloseToSource) {
    auto verify_data = bm->generateInputData();

    fft::fourier_transform_gold(false, LOG_FFT_SIZE, data->data);
    fft::fourier_transform_gold(true, LOG_FFT_SIZE, data->data);

    // Normalize iFFT result
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        data->data[i] /= (1 << LOG_FFT_SIZE);
    }

    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data->data[i]), std::abs(verify_data->data[i]), 0.001);
    }
}