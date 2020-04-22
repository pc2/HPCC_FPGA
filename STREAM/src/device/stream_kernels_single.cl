/*
STREAM kernels a single kernel combining all operations.
 They can be selected using the operation_type switch.

KERNEL_NUMBER will be replaced by the build script with the ID of the current replication.
 That means the kernels will be named copy_0, copy_1, ... up to the number of given replications.
*/
#include "parameters.h"

__kernel
__attribute__((uses_global_work_offset(0)))
void calc_KERNEL_NUMBER(__global const DEVICE_ARRAY_DATA_TYPE *restrict in1,
          __global const DEVICE_ARRAY_DATA_TYPE *restrict in2,
          __global DEVICE_ARRAY_DATA_TYPE *restrict out,
          const DEVICE_SCALAR_DATA_TYPE scalar,
          const uint array_size,
          const uint operation_type) {
#ifndef INNER_LOOP_BUFFERS
        DEVICE_ARRAY_DATA_TYPE buffer1[BUFFER_SIZE];
#endif
    uint number_elements = array_size / VECTOR_COUNT;
    for(uint i = 0;i<number_elements;i += BUFFER_SIZE){
#ifdef INNER_LOOP_BUFFERS
        DEVICE_ARRAY_DATA_TYPE buffer1[BUFFER_SIZE];
#endif
        // Load first array into buffer
        __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
        for (uint k = 0;k<BUFFER_SIZE;k++) {
            buffer1[k] = scalar * in1[i + k];
        }
        // optionally load second array into buffer for add and triad
        if (operation_type == ADD_KERNEL_TYPE || operation_type == TRIAD_KERNEL_TYPE) {
            __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
            for (uint k = 0;k<BUFFER_SIZE;k++) {
                buffer1[k] += in2[i + k];
            }
        }
        // Calculate result and write back to output array depending on chosen operation type
        __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
        for (uint k = 0;k<BUFFER_SIZE;k++) {
            out[i + k] = buffer1[k];
                   
	}
    }
}
