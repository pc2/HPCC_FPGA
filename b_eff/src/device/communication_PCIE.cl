/*
Copyright (c) 2022 Marius Meyer

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

#include "parameters.h"


/**
 *   Minimal kernel only used to measure the startup latency of a kernel and to provide a 
 *      memory buffe for Xilinx FPGAs to measure PCIe read and write performance
 *
 * @param output Output buffer that will be used to write the verification data into
 * @param verification Verification value that will be written to the buffer
 * @param messageSize size of the output buffer
 */
__kernel
__attribute__ ((max_global_work_dim(0)))
void dummyKernel(__global DEVICE_DATA_TYPE *output, DEVICE_DATA_TYPE verification, int messageSize) {
    for (int m=0; m < messageSize; m++) {
        output[m] = verification;
    }
}
