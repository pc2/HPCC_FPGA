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

#include "parameters.h"

/*
Constant used to update the pseudo random number
*/
#define POLY 7

/*
Kernel, that will update the given data array accoring to a predefined pseudo-
random access scheme. The overall data array might be equally split between
multiple kernels. In that case, the index of the current split can be given
to the kernel.

@param data The data array that will be updated
@param m The size of the data array
@param data_chunk The chunk index that has to be updated by the kernel
*/
__attribute__((max_global_work_dim(0)))
__kernel
void accessMemory_KERNEL_NUMBER(__global DEVICE_DATA_TYPE_UNSIGNED  volatile * restrict data,
                        const DEVICE_DATA_TYPE_UNSIGNED m,
                        const DEVICE_DATA_TYPE_UNSIGNED data_chunk,
                        const uint kernel_number) {
    // Initiate the pseudo random number
    DEVICE_DATA_TYPE_UNSIGNED ran = 1;

    // calculate the start of the address range this kernel is responsible for
    #ifndef SINGLE_KERNEL
    DEVICE_DATA_TYPE_UNSIGNED const address_start = kernel_number * data_chunk;
    #endif

    DEVICE_DATA_TYPE_UNSIGNED const mupdate = 4 * m;

#ifdef INTEL_FPGA
#ifndef USE_SVM
#pragma ivdep
#endif
#endif
    // do random accesses
    for (DEVICE_DATA_TYPE_UNSIGNED i=0; i< mupdate / BUFFER_SIZE; i++) {

        DEVICE_DATA_TYPE_UNSIGNED local_address[BUFFER_SIZE];
        DEVICE_DATA_TYPE_UNSIGNED loaded_data[BUFFER_SIZE];
        DEVICE_DATA_TYPE_UNSIGNED update_val[BUFFER_SIZE];

        // calculate next addresses
        __attribute__((opencl_unroll_hint(1)))
        for (int ld=0; ld< BUFFER_SIZE; ld++) {
            DEVICE_DATA_TYPE v = 0;
            if (((DEVICE_DATA_TYPE) ran) < 0) {
                v = POLY;
            }
            ran = (ran << 1) ^ v;
            update_val[ld] = ran;
            DEVICE_DATA_TYPE_UNSIGNED address = (ran >> 3) & (m - 1);
            #ifndef SINGLE_KERNEL
            local_address[ld] = address - address_start;
            #else
            local_address[ld] = address;
            #endif
#ifndef COMBINE_LOOPS
        }

        // load the data of the calculated addresses from global memory
        __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
        for (int ld=0; ld< BUFFER_SIZE; ld++) {
#endif
            #ifdef SINGLE_KERNEL
            loaded_data[ld] = data[local_address[ld]];
            #else
            if (local_address[ld] < data_chunk) {
                loaded_data[ld] = data[local_address[ld]];
            }
            #endif
        }

        // store back the calculated addresses from global memory
        __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
        for (int ld=0; ld< BUFFER_SIZE; ld++) {
            #ifdef SINGLE_KERNEL
            data[local_address[ld]] = loaded_data[ld] ^update_val[ld];
            #else
            if (local_address[ld] < data_chunk) {
                data[local_address[ld]] = loaded_data[ld] ^ update_val[ld];
            }
            #endif
        }
    }
}
