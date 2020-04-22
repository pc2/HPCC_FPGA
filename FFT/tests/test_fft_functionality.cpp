//
// Created by Marius Meyer on 20.01.20.
//

#include "gtest/gtest.h"
#include "../src/host/fft_functionality.hpp"
#include "parameters.h"



/**
 * Check if data generator generates reproducable inputs
 */
TEST (FPGASetup, DataInputReproducible) {
    auto *data1 = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    auto *data2 = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    generateInputData(data1, 1);
    generateInputData(data2, 1);
    for (int i=0; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(data1[i].real(), data2[i].real());
        EXPECT_FLOAT_EQ(data1[i].imag(), data2[i].imag());
    }
    delete [] data1;
    delete [] data2;
}

/**
 * Check if FFT of zeros returns just zeros
 */
TEST (FPGASetup, FFTReturnsZero) {
    auto *data = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(0.0);
        data[i].imag(0.0);
    }
    fourier_transform_gold(false, LOG_FFT_SIZE, data);
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        EXPECT_FLOAT_EQ(std::abs(data[i]), 0.0);
    }
    delete [] data;
}


/**
 * Check if FFT calculates the correct result for all number being 1.0,1.0i
 */
TEST (FPGASetup, FFTCloseToZeroForAll1And1) {
    auto *data = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(1.0);
        data[i].imag(1.0);
    }
    fourier_transform_gold(false, LOG_FFT_SIZE, data);
    EXPECT_NEAR(data[0].real(), (1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data[0].imag(), (1 << LOG_FFT_SIZE), 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data[i].imag(), 0.0, 0.00001);
    }
    delete [] data;
}

/**
* Check if iFFT calculates the correct result for all number being 1.0,1.0i
*/
TEST (FPGASetup, IFFTCloseToZeroForAll1And1) {
    auto *data = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    for (int i=0; i<(1 << LOG_FFT_SIZE); i++) {
        data[i].real(1.0);
        data[i].imag(1.0);
    }
    fourier_transform_gold(true, LOG_FFT_SIZE, data);
    EXPECT_NEAR(data[0].real(), (1 << LOG_FFT_SIZE), 0.00001);
    EXPECT_NEAR(data[0].imag(), (1 << LOG_FFT_SIZE), 0.00001);
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(data[i].real(), 0.0, 0.00001);
        EXPECT_NEAR(data[i].imag(), 0.0, 0.00001);
    }
    delete [] data;
}


/**
 * Check if FFT and FFT check give low error when FFT is calculated directly
 */
TEST (FPGASetup, FFTandiFFTProduceResultCloseToSource) {
    auto *data = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    auto *verify_data = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    generateInputData(data, 1);
    generateInputData(verify_data, 1);

    fourier_transform_gold(false, LOG_FFT_SIZE, data);
    fourier_transform_gold(true, LOG_FFT_SIZE, data);

    // Normalize iFFT result
    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        data[i] /= (1 << LOG_FFT_SIZE);
    }

    for (int i=1; i < (1 << LOG_FFT_SIZE); i++) {
        EXPECT_NEAR(std::abs(data[i]), std::abs(verify_data[i]), 0.001);
    }
    delete [] data;
    delete [] verify_data;
}