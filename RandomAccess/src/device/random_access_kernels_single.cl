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

#define CONCURRENT_GEN_LOG HPCC_FPGA_RA_RNG_COUNT_LOG
#define CONCURRENT_GEN (1 << CONCURRENT_GEN_LOG)
#define SHIFT_GAP HPCC_FPGA_RA_RNG_DISTANCE

#define BLOCK_SIZE_LOG GLOBAL_MEM_UNROLL_LOG
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG)

/* PY_CODE_GEN 
try:
    kernel_param_attributes = generate_attributes(num_replications)
except:
    kernel_param_attributes = ["" for i in range(num_replications)]
*/

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]

/*
Kernel, that will update the given data array accoring to a predefined pseudo-
random access scheme. The overall data array might be equally split between
multiple kernels. In that case, the index of the current split can be given
to the kernel.

@param data The data array that will be updated
@param m  the size of the data array
@param data_chunk  the chunk size that has to be updated by the kernel
@param kernel_number Number of the kernel that defines the offset of the data chunk to the total data array
*/
__attribute__((max_global_work_dim(0),uses_global_work_offset(0)))
__kernel
void accessMemory_/*PY_CODE_GEN i*/(__global /*PY_CODE_GEN kernel_param_attributes[i]*/ DEVICE_DATA_TYPE_UNSIGNED  volatile * restrict data,
                        __constant /*PY_CODE_GEN kernel_param_attributes[i]*/ const DEVICE_DATA_TYPE_UNSIGNED * restrict random_init,
                        const DEVICE_DATA_TYPE_UNSIGNED m,
                        const DEVICE_DATA_TYPE_UNSIGNED data_chunk,
                        const uint num_cache_operations,
                        const uint kernel_number) {

    // Initiate the pseudo random number generators
    DEVICE_DATA_TYPE_UNSIGNED ran_initials[CONCURRENT_GEN/BLOCK_SIZE][BLOCK_SIZE];
    // Load RNG initial values from global memory
    for (int r = 0; r < CONCURRENT_GEN/BLOCK_SIZE; r++) {
        DEVICE_DATA_TYPE_UNSIGNED tmp[BLOCK_SIZE];
         __attribute__((opencl_unroll_hint(BLOCK_SIZE)))
        for (int b = 0; b < BLOCK_SIZE; b++) {
            tmp[b] = random_init[r* BLOCK_SIZE + b];
        }
        __attribute__((opencl_unroll_hint(BLOCK_SIZE)))
        for (int b = 0; b < BLOCK_SIZE; b++) {
            ran_initials[r][b] = tmp[b];
        }
    }
    DEVICE_DATA_TYPE_UNSIGNED ran[CONCURRENT_GEN];
    DEVICE_DATA_TYPE_UNSIGNED number_count[CONCURRENT_GEN];
    __attribute__((opencl_unroll_hint(CONCURRENT_GEN)))
    for (int r = 0; r < CONCURRENT_GEN; r++) {
        number_count[r] = 0;
        ran[r] = ran_initials[r >> BLOCK_SIZE_LOG][ r & (BLOCK_SIZE - 1)];
    }

    // Initialize shift register
    // this is the data shift register that contains the random numbers
    DEVICE_DATA_TYPE_UNSIGNED random_number_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];
    // these two shift registers contain a valid bit and a complete bit
    // the valid bit is set, if the current random number is valid and in the range of the current kernel
    // the complete bit is only set if all random number generators before the current one have completed execution
    bool random_number_valid[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];
    bool random_number_done_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];

 __attribute__((opencl_unroll_hint((CONCURRENT_GEN + 1) * SHIFT_GAP + 1)))
    for (int r = 0; r < (CONCURRENT_GEN + 1) * SHIFT_GAP + 1; r++) {
        random_number_shift[r] = 0;
        random_number_done_shift[r] = false;
        random_number_valid[r] = false;
    }

    // calculate the start of the address range this kernel is responsible for
    #ifndef SINGLE_KERNEL
    DEVICE_DATA_TYPE_UNSIGNED const address_start = kernel_number * data_chunk;
    #endif

    DEVICE_DATA_TYPE_UNSIGNED const mupdate = 4 * m;
    bool done = false;

    // do random accesses until we achieved the desired number of updates
#ifdef INTEL_FPGA
#ifdef HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP
#pragma ivdep array(data)
#endif
#endif
    while (!done) {

        DEVICE_DATA_TYPE_UNSIGNED local_address_buffer[BUFFER_SIZE];
        DEVICE_DATA_TYPE_UNSIGNED loaded_data_buffer[BUFFER_SIZE];
        DEVICE_DATA_TYPE_UNSIGNED update_val_buffer[BUFFER_SIZE];

        __attribute__((opencl_unroll_hint(1)))
        for (uint i = 0; i < BUFFER_SIZE; i++) {
            // Fully unrolled loop resembling the RNGs
            // They put a new random number into the shift register if it does not already contain a valid random number indicated by the valid bit
            __attribute__((opencl_unroll_hint(CONCURRENT_GEN)))
            for (int r=0; r < CONCURRENT_GEN; r++) {
                DEVICE_DATA_TYPE_UNSIGNED total_updates = (mupdate >> CONCURRENT_GEN_LOG) + ((r < (mupdate & (CONCURRENT_GEN - 1)) ? 1 : 0));
                number_count[r] = !random_number_valid[(r + 1) * SHIFT_GAP] ? number_count[r] + 1 : number_count[r];
                bool is_inrange = false;
                if (!random_number_valid[(r + 1) * SHIFT_GAP] && number_count[r] <= total_updates) {
                    DEVICE_DATA_TYPE_UNSIGNED v = ((DEVICE_DATA_TYPE) ran[r] < 0) ? POLY : 0UL;
                    ran[r] = (ran[r] << 1) ^ v;
                    DEVICE_DATA_TYPE_UNSIGNED address = (ran[r] >> 3) & (m - 1);
                    #ifndef SINGLE_KERNEL
                    DEVICE_DATA_TYPE_UNSIGNED local_address = address - address_start;
                    #else
                    DEVICE_DATA_TYPE_UNSIGNED local_address = address;
                    #endif
                    is_inrange = (local_address < data_chunk);
                    random_number_shift[(r + 1) * SHIFT_GAP] = ran[r];
                    // printf("Update: %d, address: %lu, with: %lu\n", (local_address < data_chunk), local_address, ran[r]);
                }
                // update the status bits of the shift register accordingly
                random_number_valid[(r + 1) * SHIFT_GAP] = (random_number_valid[(r + 1) * SHIFT_GAP] || is_inrange);
                random_number_done_shift[(r + 1) * SHIFT_GAP] = (number_count[r] >= total_updates && (random_number_done_shift[(r + 1) * SHIFT_GAP] || r == CONCURRENT_GEN - 1));
            }

            // Get random number from shift register and do update
            DEVICE_DATA_TYPE_UNSIGNED random_number = random_number_shift[0];
            bool valid = random_number_valid[0];
            done = random_number_done_shift[0];
            DEVICE_DATA_TYPE_UNSIGNED address = (random_number >> 3) & (m - 1);
            #ifndef SINGLE_KERNEL
            DEVICE_DATA_TYPE_UNSIGNED local_address = address - address_start;
            #else
            DEVICE_DATA_TYPE_UNSIGNED local_address = address;
            #endif

            local_address_buffer[i] = local_address;

            if (valid) {
                loaded_data_buffer[i] = data[local_address] ^ random_number;
            }

            // Shift the contents of the shift register
            __attribute__((opencl_unroll_hint((CONCURRENT_GEN + 1) * SHIFT_GAP)))
            for (int r = 0; r < (CONCURRENT_GEN + 1) * SHIFT_GAP; r++) {
                random_number_shift[r] = random_number_shift[r + 1];
                random_number_done_shift[r] = random_number_done_shift[r + 1];
                random_number_valid[r] = random_number_valid[r + 1];
            }
            // Set the last value in the shift register to invalid so the RNGs can update it
            random_number_valid[(CONCURRENT_GEN + 1) * SHIFT_GAP] = false;
            random_number_done_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP] = false;
        }

        __attribute__((opencl_unroll_hint(1)))
        for (uint i = 0; i < BUFFER_SIZE; i++) {
            DEVICE_DATA_TYPE_UNSIGNED local_address = local_address_buffer[i];
            if (local_address < data_chunk) {
                data[local_address] = loaded_data_buffer[i];
            }
        }
    }
}

// PY_CODE_GEN block_end
