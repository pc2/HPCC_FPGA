//
// Created by Marius Meyer on 04.12.19.
//

/*
Copyright (c) 2019 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "fft_functionality.hpp"

/* C++ standard library headers */
#include <memory>
#include <random>

/* Project's headers */
#include "execution.h"
#include "cxxopts.hpp"
#include "setup/fpga_setup.hpp"
#include "setup/common_benchmark_io.hpp"
#include "parameters.h"


/**
Prints the execution results to stdout

@param results The execution results
*/
void
printResults(std::shared_ptr<bm_execution::ExecutionTimings> results) {

    double gflop = 5 * (1 << LOG_FFT_SIZE) * LOG_FFT_SIZE * results->iterations * 1.0e-9;

    double minTime = *min_element(results->calculationTimings.begin(), results->calculationTimings.end());
    double avgTime = accumulate(results->calculationTimings.begin(), results->calculationTimings.end(), 0.0)
                                                                                / results->calculationTimings.size();

    std::cout << std::setw(ENTRY_SPACE) << " " << std::setw(ENTRY_SPACE) << "avg"
              << std::setw(ENTRY_SPACE) << "best" << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << "Time in s:" << std::setw(ENTRY_SPACE) << avgTime / results->iterations
                << std::setw(ENTRY_SPACE) << minTime / results->iterations << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << "GFLOPS:" << std::setw(ENTRY_SPACE) << gflop / avgTime
                << std::setw(ENTRY_SPACE) << gflop / minTime << std::endl;

}


void generateInputData(std::complex<HOST_DATA_TYPE>* data, unsigned iterations) {
    std::mt19937 gen(0);
    auto dis = std::uniform_real_distribution<HOST_DATA_TYPE>(-1.0, 1.0);
    for (int i=0; i< iterations * (1 << LOG_FFT_SIZE); i++) {
        data[i].real(dis(gen));
        data[i].imag(dis(gen));
    }
}

double checkFFTResult(std::complex<HOST_DATA_TYPE>* verify_data, std::complex<HOST_DATA_TYPE>* result_data, unsigned iterations) {
    double residual_max = 0;
    for (int i = 0; i < iterations; i++) {
        // we have to bit reverse the output data of the FPGA kernel, since it will be provided in bit-reversed order.
        // Directly applying iFFT on the data would thus not form the identity function we want to have for verification.
        // TODO: This might need to be changed for other FPGA implementations that return the data in correct order
        bit_reverse(&result_data[i * (1 << LOG_FFT_SIZE)], 1);
        fourier_transform_gold(true, LOG_FFT_SIZE, &result_data[i * (1 << LOG_FFT_SIZE)]);

        // Normalize the data after applying iFFT
        for (int j = 0; j < (1 << LOG_FFT_SIZE); j++) {
            result_data[i * (1 << LOG_FFT_SIZE) + j] /= (1 << LOG_FFT_SIZE);
        }
        for (int j = 0; j < (1 << LOG_FFT_SIZE); j++) {
            double tmp_error =  std::abs(verify_data[i * (1 << LOG_FFT_SIZE) + j] - result_data[i * (1 << LOG_FFT_SIZE) + j]);
            residual_max = residual_max > tmp_error ? residual_max : tmp_error;
        }
    }
    double error = residual_max /
                   (std::numeric_limits<HOST_DATA_TYPE>::epsilon() * LOG_FFT_SIZE);

    std::cout << std::setw(ENTRY_SPACE) << "res. error" << std::setw(ENTRY_SPACE) << "mach. eps" << std::endl;
    std::cout << std::setw(ENTRY_SPACE) << error << std::setw(ENTRY_SPACE)
              << std::numeric_limits<HOST_DATA_TYPE>::epsilon() << std::endl << std::endl;

    // Calculate residual according to paper considering also the used iterations
    return error;
}

void bit_reverse(std::complex<HOST_DATA_TYPE> *data, unsigned iterations) {
    auto *tmp = new std::complex<HOST_DATA_TYPE>[(1 << LOG_FFT_SIZE)];
    for (int k=0; k < iterations; k++) {
        for (int i = 0; i < (1 << LOG_FFT_SIZE); i++) {
            int fwd = i;
            int bit_rev = 0;
            for (int j = 0; j < LOG_FFT_SIZE; j++) {
                bit_rev <<= 1;
                bit_rev |= fwd & 1;
                fwd >>= 1;
            }
            tmp[i] = data[bit_rev];
        }
        for (int i = 0; i < (1 << LOG_FFT_SIZE); i++) {
            data[k * (1 << LOG_FFT_SIZE) + i] = tmp[i];
        }
    }
    delete [] tmp;
}

// The function definitions and implementations below this comment are taken from the
// FFT1D example implementation of the Intel FPGA SDK for OpenCL 19.4
// They are licensed under the following conditions:
//
// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.


void fourier_transform_gold(bool inverse, const int lognr_points, std::complex<HOST_DATA_TYPE> *data_sp) {
    const int nr_points = 1 << lognr_points;

    auto *data = new std::complex<double>[nr_points];

    for (int i=0; i < nr_points; i++) {
        data[i] = data_sp[i];
    }

    // The inverse requires swapping the real and imaginary component

    if (inverse) {
        for (int i = 0; i < nr_points; i++) {
            double tmp = data[i].imag();
            data[i].imag(data[i].real());
            data[i].real(tmp);
        }
    }
    // Do a FT recursively
    fourier_stage(lognr_points, data);

    // The inverse requires swapping the real and imaginary component
    if (inverse) {
        for (int i = 0; i < nr_points; i++) {
            double tmp = data[i].real();
            data[i].real(data[i].imag());
            data[i].imag(tmp);
        }
    }

    // Do the bit reversal
    for (int i = 0; i < nr_points; i++) {
        data_sp[i] = data[i];
    }

    delete [] data;
}

void fourier_stage(int lognr_points, std::complex<double> *data) {
    int nr_points = 1 << lognr_points;
    if (nr_points == 1) return;
    auto *half1 = new std::complex<double>[nr_points / 2];
    auto *half2 = new std::complex<double>[nr_points / 2];
    for (int i = 0; i < nr_points / 2; i++) {
        half1[i] = data[2 * i];
        half2[i] = data[2 * i + 1];
    }
    fourier_stage(lognr_points - 1, half1);
    fourier_stage(lognr_points - 1, half2);
    for (int i = 0; i < nr_points / 2; i++) {
        data[i].real(half1[i].real() + cos (2 * M_PI * i / nr_points) * half2[i].real() + sin (2 * M_PI * i / nr_points) * half2[i].imag());
        data[i].imag(half1[i].imag() - sin (2 * M_PI * i / nr_points) * half2[i].real() + cos (2 * M_PI * i / nr_points) * half2[i].imag());
        data[i + nr_points / 2].real(half1[i].real() - cos (2 * M_PI * i / nr_points) * half2[i].real() - sin (2 * M_PI * i / nr_points) * half2[i].imag());
        data[i + nr_points / 2].imag(half1[i].imag() + sin (2 * M_PI * i / nr_points) * half2[i].real() - cos (2 * M_PI * i / nr_points) * half2[i].imag());
    }

    delete [] half1;
    delete [] half2;
}