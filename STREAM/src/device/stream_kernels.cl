/*
STREAM kernels as separate kernels for each operation.

KERNEL_NUMBER will be replaced by the build script with the ID of the current replication.
 That means the kernels will be named copy_0, copy_1, ... up to the number of given replications.
*/
#include "parameters.h"

{% for i in range(num_replications) %}

__kernel
__attribute__((uses_global_work_offset(0)))
void copy_{{ i }}(__global const DEVICE_ARRAY_DATA_TYPE * restrict in,
          __global DEVICE_ARRAY_DATA_TYPE * restrict out,
          const uint array_size) {
    uint number_elements = array_size / VECTOR_COUNT;
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
    for(uint i = 0; i < number_elements; i++){
        out[i] = in[i];
    }
}

__kernel
__attribute__((uses_global_work_offset(0)))
void add_{{ i }}(__global const DEVICE_ARRAY_DATA_TYPE * restrict in1,
          __global const DEVICE_ARRAY_DATA_TYPE * restrict in2,
          __global DEVICE_ARRAY_DATA_TYPE * restrict out,
          const uint array_size) {
    uint number_elements = array_size / VECTOR_COUNT;
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
    for (uint i=0; i<number_elements; i++){
        out[i] = in1[i] + in2[i];
    }
}

__kernel
__attribute__((uses_global_work_offset(0)))
void scale_{{ i }}(__global const DEVICE_ARRAY_DATA_TYPE * restrict in,
          __global DEVICE_ARRAY_DATA_TYPE * restrict out,
          const DEVICE_SCALAR_DATA_TYPE scalar,
          const uint array_size) {
    uint number_elements = array_size / VECTOR_COUNT;
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
    for (uint i=0; i<number_elements; i++){
        out[i] = scalar * in[i];
    }
}

__kernel
__attribute__((uses_global_work_offset(0)))
void triad_{{ i }}(__global const DEVICE_ARRAY_DATA_TYPE * restrict in1,
          __global const DEVICE_ARRAY_DATA_TYPE * restrict in2,
          __global DEVICE_ARRAY_DATA_TYPE * restrict out,
          const DEVICE_SCALAR_DATA_TYPE scalar,
          const uint array_size) {
    uint number_elements = array_size / VECTOR_COUNT;
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
    for (uint i=0; i<number_elements; i++){
        out[i] = in1[i] + scalar * in2[i];
    }
}

{% endfor %}