/*
This file contains the OpenCL implementation of all four STREAM operations in a single kernel.
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
#if (BUFFER_SIZE > UNROLL_COUNT)
// Disable pipelining of the outer loop for Intel FPGA.
// Only pipeline outer loop, if the inner loops are fully unrolled
#pragma disable_loop_pipelining
#endif
#endif
    // Process every element in the global memory arrays by loading chunks of data
    // that fit into the local memory buffer
    for(uint i = 0;i<number_elements;i += BUFFER_SIZE){
#ifdef INNER_LOOP_BUFFERS
        DEVICE_ARRAY_DATA_TYPE buffer1[BUFFER_SIZE];
#endif
#ifdef INTEL_FPGA
// Disable fusion of loops, since they are meant to be executed sequentially
#pragma nofusion
#endif
        // Load chunk of first array into buffer and scale the values
        for (uint k = 0;k<BUFFER_SIZE; k += UNROLL_COUNT) {
            // Registers used to store the values for all unrolled
            // load operations from global memory
            DEVICE_ARRAY_DATA_TYPE chunk[UNROLL_COUNT];

            // Load values from global memory into the registers
            // The number of values is defined by UNROLL_COUNT
            __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
            for (uint u = 0; u < UNROLL_COUNT; u++) {
                chunk[u] = in1[i + k + u];
            }

            // Scale the values in the registers and store the
            // result in the local memory buffer
            __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
            for (uint u = 0; u < UNROLL_COUNT; u++) {
                buffer1[k + u] = scalar * chunk[u];
            }
        }
        // optionally load chunk of second array into buffer for add and triad
        if (operation_type == ADD_KERNEL_TYPE || operation_type == TRIAD_KERNEL_TYPE) {
#ifdef INTEL_FPGA
// Disable fusion of loops, since they are meant to be executed sequentially
#pragma nofusion
#endif
            for (uint k = 0;k<BUFFER_SIZE; k += UNROLL_COUNT) {
                // Registers used to store the values for all unrolled
                // load operations from global memory
                DEVICE_ARRAY_DATA_TYPE chunk[UNROLL_COUNT];

                // Load values from global memory into the registers
                // The number of values is defined by UNROLL_COUNT
                __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                for (uint u = 0; u < UNROLL_COUNT; u++) {
                    chunk[u] = in2[i + k + u];
                }

                // Add the values in the registers to the
                // values stored in local memory
                __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                for (uint u = 0; u < UNROLL_COUNT; u++) {
                    buffer1[k + u] += chunk[u];
                }
            }
        }
        
        // Read the cumputed chunk of the output array from local memory
        // and store it in global memory
#ifdef INTEL_FPGA
// Disable fusion of loops, since they are meant to be executed sequentially
#pragma nofusion
#endif
        for (uint k = 0;k<BUFFER_SIZE; k += UNROLL_COUNT) {
            // Registers used to store the values for all unrolled
            // load operations from local memory
            DEVICE_ARRAY_DATA_TYPE chunk[UNROLL_COUNT];

            // Load values from local memory into the registers
            // The number of values is defined by UNROLL_COUNT
            __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
            for (uint u = 0; u < UNROLL_COUNT; u++) {
                chunk[u] = buffer1[k + u];
            }

            // Store the values in the registers in global memory
            __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
            for (uint u = 0; u < UNROLL_COUNT; u++) {
                out[i + k + u] = chunk[u];  
            }             
    	}
    }
}

// PY_CODE_GEN block_end
