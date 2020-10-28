/*
STREAM kernels a single kernel combining all operations.
 They can be selected using the operation_type switch.

KERNEL_NUMBER will be replaced by the build script with the ID of the current replication.
 That means the kernels will be named copy_0, copy_1, ... up to the number of given replications.
*/
#include "parameters.h"

#if DATA_TYPE_SIZE == 8
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if DATA_TYPE_SIZE == 2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

/* PY_CODE_GEN 
try:
    kernel_param_attributes = generate_attributes(num_replications)
except:
    kernel_param_attributes = ["" for i in range(num_replications)]
*/
// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]
__kernel
__attribute__((uses_global_work_offset(0)))
void calc_/*PY_CODE_GEN i*/(__global /*PY_CODE_GEN kernel_param_attributes[i]*/ const DEVICE_ARRAY_DATA_TYPE *restrict in1,
          __global /*PY_CODE_GEN kernel_param_attributes[i]*/ const DEVICE_ARRAY_DATA_TYPE *restrict in2,
          __global /*PY_CODE_GEN kernel_param_attributes[i]*/ DEVICE_ARRAY_DATA_TYPE *restrict out,
          const DEVICE_SCALAR_DATA_TYPE scalar,
          const uint array_size,
          const uint operation_type) {
#ifndef INNER_LOOP_BUFFERS
        DEVICE_ARRAY_DATA_TYPE buffer1[BUFFER_SIZE];
#endif
    uint number_elements = array_size / VECTOR_COUNT;
#ifdef INTEL_FPGA
#pragma disable_loop_pipelining
#endif
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

// PY_CODE_GEN block_end
