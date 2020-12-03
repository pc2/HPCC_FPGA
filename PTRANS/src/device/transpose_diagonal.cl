/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/

#include "parameters.h"

// Need some depth to our channels to accommodate their bursty filling.
#ifdef INTEL_FPGA
#pragma OPENCL EXTENSION cl_intel_channels : enable

struct ch_data {
    DEVICE_DATA_TYPE data[CHANNEL_WIDTH];
};

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_total_replications)]
// Channel used to send the transposed blocks of A
channel ch_data chan_a_out/*PY_CODE_GEN i*/[GLOBAL_MEM_UNROLL/CHANNEL_WIDTH] __attribute__((depth(POINTS)));
channel ch_data chan_a_in/*PY_CODE_GEN i*/[GLOBAL_MEM_UNROLL/CHANNEL_WIDTH] __attribute__((depth(POINTS)));
// PY_CODE_GEN block_end
#endif



/**
* Load a block of A into local memory in a reordered fashion
* to transpose it half-way
*
*
* @param A Buffer for matrix A
* @param local_buffer The local memory buffer the block is stored into
* @param current_block Index of the current block used to calculate the offset in global memory
*
*/
void
load_chunk_of_a(__global DEVICE_DATA_TYPE *restrict A,
        DEVICE_DATA_TYPE[BLOCK_SIZE][BLOCK_SIZE] local_buffer,
        const int current_block,
        const int row,
        const int col) {

        unsigned local_mem_converted_row = row * (BLOCK_SIZE / GLOBAL_MEM_UNROLL) + col;

        DEVICE_DATA_TYPE rotate_in[GLOBAL_MEM_UNROLL];

        // Blocks of a will be stored columnwise in global memory
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
        for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
            rotate_in[unroll_count] = A[block * BLOCK_SIZE * BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count +
                                                                    row * BLOCK_SIZE];
        }

        unsigned rot = row & (GLOBAL_MEM_UNROLL - 1);

        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
        for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
            // every block of (N / GLOBAL_MEM_UNROLL), rotates the index by 1
            // store in double buffer
            local_buffer[local_mem_converted_row][unroll_count] = rotate_in[(unroll_count + GLOBAL_MEM_UNROLL - rot)
                                                                                        & (GLOBAL_MEM_UNROLL - 1)];
        }
}

/**
* send a chunk of A into local memory in a reordered fashion
* to transpose it half-way
*
*
* @param A Buffer for matrix A
* @param local_buffer The local memory buffer the block is stored into
* @param current_block Index of the current block used to calculate the offset in global memory
*
*/
void
send_chunk_of_a(__global DEVICE_DATA_TYPE *restrict A,
        DEVICE_DATA_TYPE[BLOCK_SIZE][BLOCK_SIZE] local_buffer,
        const int current_block,
        const int row,
        const int col) {

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

        DEVICE_DATA_TYPE channel_data[GLOBAL_MEM_UNROLL / CHANNEL_WIDTH][GLOBAL_MEM_UNROLL & (CHANNEL_WIDTH - 1)];
        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
        for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {
            channel_data[unroll_count / CHANNEL_WIDTH][unroll_count & (CHANNEL_WIDTH - 1)] = rotate_out[(unroll_count + rot_out) & (GLOBAL_MEM_UNROLL - 1)];
        }

__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL/CHANNEL_WIDTH)))
        for (unsigned c = 0; c < GLOBAL_MEM_UNROLL/CHANNEL_WIDTH; c++) {
            write_channel_intel(chan_a_out/*PY_CODE_GEN i*/, channel_data[c]); 
        }
}

/**
 * Read blocks of matrix A and transpose them in memory.
 * Write the block into an external channel.
 *
 * Will do the following:
 *
 * A -> trans(A) -> ext. ch
 *
 * @param A Buffer for matrix A
 * @param matrixSize Size of the matrices. Must be multiple of BLOCK_SIZE
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose_read(__global DEVICE_DATA_TYPE *restrict A,
            const uint number_of_blocks) {

    // transpose the matrix block-wise from global memory
    // One extra iteration to empty double buffer
    for (int block = 0; block < number_of_blocks + 1; block++) {

        // local memory double buffer for a matrix block
        DEVICE_DATA_TYPE a_block[2][BLOCK_SIZE * BLOCK_SIZE / GLOBAL_MEM_UNROLL][GLOBAL_MEM_UNROLL] __attribute__((xcl_array_partition(cyclic, GLOBAL_MEM_UNROLL,1))) __attribute__((xcl_array_partition(cyclic, GLOBAL_MEM_UNROLL,2)));

        // read in block from global memory and store it in a memory efficient manner
#pragma loop_coalesce 2
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < BLOCK_SIZE / GLOBAL_MEM_UNROLL; col++) {
                load_chunk_of_a(A, a_block[block & 1], block, row, col);
                send_chunk_of_a(a_block[block & 1], block, row, col);
            }
        }
    }
}

/**
 * Will add a matrix from external channel and matrix from global memory and store result in global memory.
 *
 * Will do the following:
 *
 * ext. ch + B --> A_out
 *
 * where A_out, ext. ch and B are matrices of size matrixSize*matrixSize
 *
 * @param B Buffer for matrix B
 * @param A_out Output buffer for result matrix
 * @param matrixSize Size of the matrices. Must be multiple of BLOCK_SIZE
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose_write(__global DEVICE_DATA_TYPE *restrict B,
            __global DEVICE_DATA_TYPE *restrict A_out,
            const uint number_of_blocks) {

    for (int block = 0; block < number_of_blocks; block++) {
        // complete matrix transposition and write the result back to global memory
#pragma loop_coalesce 2
        for (int row = 0; row < BLOCK_SIZE; row++) {
            for (int col = 0; col < BLOCK_SIZE / GLOBAL_MEM_UNROLL; col++) {

                DEVICE_DATA_TYPE channel_data[GLOBAL_MEM_UNROLL / CHANNEL_WIDTH][GLOBAL_MEM_UNROLL & (CHANNEL_WIDTH - 1)];

                __attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL/CHANNEL_WIDTH)))
                for (unsigned c = 0; c < GLOBAL_MEM_UNROLL/CHANNEL_WIDTH; c++) {
                     channel_data[c] = read_channel_intel(chan_a_out/*PY_CODE_GEN i*/); 
                }

                unsigned rot_out = row & (GLOBAL_MEM_UNROLL - 1);
                // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                for (unsigned unroll_count = 0; unroll_count < GLOBAL_MEM_UNROLL; unroll_count++) {

                    A_out[block * BLOCK_SIZE * BLOCK_SIZE +
                    row * BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count] =
                    channel_data[unroll_count / CHANNEL_WIDTH][unroll_count & (CHANNEL_WIDTH - 1)]
                    + B[block * BLOCK_SIZE * BLOCK_SIZE +
                    row * BLOCK_SIZE + col * GLOBAL_MEM_UNROLL + unroll_count];
                }
            }
        }
    }
}
