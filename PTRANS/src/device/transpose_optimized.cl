/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/

#include "parameters.h"

/**
 * Optimized matrix transposition that simplifies local memory accesses.
 *
 * Will do the following:
 *
 * A_out = trans(A) + B
 *
 * where A_out, A and B are matrices of size matrixSize*matrixSize
 *
 * @param A Buffer for matrix A
 * @param B Buffer for matrix B
 * @param A_out Output buffer for result matrix
 * @param matrixSize Size of the matrices. Must be multiple of BLOCK_SIZE
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose(__global DEVICE_DATA_TYPE *restrict A,
            __global DEVICE_DATA_TYPE *restrict B,
            __global DEVICE_DATA_TYPE *restrict A_out,
            const uint number_of_blocks) {

    const unsigned matrixSize = number_of_blocks * BLOCK_SIZE;

    // transpose the matrix block-wise from global memory
#pragma loop_coalesce 2
    for (int block_row = 0; block_row < number_of_blocks; block_row++) {
        for (int block_col = 0; block_col < number_of_blocks; block_col++) {

            // local memory buffer for a matrix block
            DEVICE_DATA_TYPE a_block[BLOCK_SIZE * BLOCK_SIZE / GLOBAL_MEM_UNROLL][GLOBAL_MEM_UNROLL] __attribute__((xcl_array_partition(cyclic, GLOBAL_MEM_UNROLL,1))) __attribute__((xcl_array_partition(cyclic, GLOBAL_MEM_UNROLL,2)));

            // read in block from global memory and store it in a memory efficient manner
#pragma loop_coalesce 2
            for (int row = 0; row < BLOCK_SIZE; row++) {
                for (int col = 0; col < BLOCK_SIZE / GLOBAL_MEM_UNROLL; col++) {

                    unsigned local_mem_converted_row = row * (BLOCK_SIZE / GLOBAL_MEM_UNROLL) + col;

                    DEVICE_DATA_TYPE rotate_in[GLOBAL_MEM_UNROLL];

__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
                        rotate_in[unroll_count] = A[block_col * BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count +
                                                                                (block_row * BLOCK_SIZE + row) * matrixSize];
                    }

                    unsigned rot = row & (GLOBAL_MEM_UNROLL - 1);

                    // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
                        // every block of (N / GLOBAL_MEM_UNROLL), rotates the index by 1
                        a_block[local_mem_converted_row][unroll_count] = rotate_in[(unroll_count + GLOBAL_MEM_UNROLL - rot)
                                                                                                    & (GLOBAL_MEM_UNROLL - 1)];
                    }
                }
            }

        // complete matrix transposition and write the result back to global memory
#pragma loop_coalesce 2
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < BLOCK_SIZE / GLOBAL_MEM_UNROLL; col++) {

                    DEVICE_DATA_TYPE rotate_out[GLOBAL_MEM_UNROLL];

                    unsigned base = col * BLOCK_SIZE;
                    unsigned offset = row / GLOBAL_MEM_UNROLL;


__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
                        unsigned rot = ((GLOBAL_MEM_UNROLL + unroll_count - row) * (BLOCK_SIZE / GLOBAL_MEM_UNROLL)) &
                                                                                                    (BLOCK_SIZE - 1);
                        unsigned row_rotate = base + offset + rot;
                        rotate_out[unroll_count] = a_block[row_rotate][unroll_count];
                    }

                    unsigned rot_out = row & (GLOBAL_MEM_UNROLL - 1);
                    // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {

                        A_out[(block_col * BLOCK_SIZE + row) * matrixSize +
                        block_row* BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count] =
                        rotate_out[(unroll_count + rot_out) & (GLOBAL_MEM_UNROLL - 1)]
                        + B[(block_col * BLOCK_SIZE + row) * matrixSize +
                        block_row * BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count];
                    }
                }
            }
        }
    }
}