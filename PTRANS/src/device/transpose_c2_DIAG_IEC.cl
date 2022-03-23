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

typedef struct {
    DEVICE_DATA_TYPE data[CHANNEL_WIDTH/2];
} ch_data;

{% for i in range(num_replications) %}
// Channel used to send the transposed blocks of A
channel ch_data chan_a_out1{{ i }} __attribute((io({{ "\"kernel_output_ch{}\"".format(2*i) }}), depth(1)));
channel ch_data chan_a_out2{{ i }} __attribute((io({{ "\"kernel_output_ch{}\"".format(2*i + 1) }}), depth(1)));
channel ch_data chan_a_in1{{ i }} __attribute((io({{ "\"kernel_input_ch{}\"".format(2*i + 1) }}), depth(1)));
channel ch_data chan_a_in2{{ i }} __attribute((io({{ "\"kernel_input_ch{}\"".format(2*i) }}), depth(1)));
{% endfor %}
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
        DEVICE_DATA_TYPE local_buffer[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
        const ulong  current_block,
        const ulong  row,
        const ulong col) {

        ulong local_mem_converted_row = row * (BLOCK_SIZE / CHANNEL_WIDTH) + col;

        DEVICE_DATA_TYPE rotate_in[CHANNEL_WIDTH];

        // Blocks of a will be stored columnwise in global memory
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            rotate_in[unroll_count] = A[current_block * BLOCK_SIZE * BLOCK_SIZE + col * CHANNEL_WIDTH + unroll_count +
                                                                    row * BLOCK_SIZE];
        }

        unsigned rot = row & (CHANNEL_WIDTH - 1);

        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            // every block of (N / CHANNEL_WIDTH), rotates the index by 1
            // store in double buffer
            local_buffer[local_mem_converted_row][unroll_count] = rotate_in[(unroll_count + CHANNEL_WIDTH - rot)
                                                                                        & (CHANNEL_WIDTH - 1)];
        }
}

{% for i in range(num_total_replications) %}

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
send_chunk_of_a{{ i }}(const DEVICE_DATA_TYPE local_buffer[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
        const ulong row,
        const ulong col) {

        DEVICE_DATA_TYPE rotate_out[CHANNEL_WIDTH];

        ulong base = col * BLOCK_SIZE;
        ulong offset = row / CHANNEL_WIDTH;


__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            unsigned rot = ((CHANNEL_WIDTH + unroll_count - row) * (BLOCK_SIZE / CHANNEL_WIDTH)) &
                                                                                        (BLOCK_SIZE - 1);
            unsigned row_rotate = base + offset + rot;
            rotate_out[unroll_count] = local_buffer[row_rotate][unroll_count];
        }

        unsigned rot_out = row & (CHANNEL_WIDTH - 1);

        DEVICE_DATA_TYPE channel_data[CHANNEL_WIDTH];
        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            channel_data[unroll_count] = rotate_out[(unroll_count + rot_out) & (CHANNEL_WIDTH - 1)];
        }

        ch_data data1;
        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH/2)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH/2; unroll_count++) {
            data1.data[unroll_count] = channel_data[unroll_count];
        }
        write_channel_intel(chan_a_out1{{ i }}, data1); 

        ch_data data2;
        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH/2)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH/2; unroll_count++) {
            data2.data[unroll_count] = channel_data[CHANNEL_WIDTH/2 + unroll_count];
        }
        write_channel_intel(chan_a_out2{{ i }}, data2); 
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
 * @param block_offset The first block that will be processed in the provided buffer
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose_read{{ i }}(__global DEVICE_DATA_TYPE *restrict A,
            const ulong block_offset,
            const ulong number_of_blocks) {

    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_block[2][BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH] __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,1))) __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,2)));

    // transpose the matrix block-wise from global memory
    // One extra iteration to empty double buffer
    #pragma loop_coalesce
    for (ulong block = block_offset; block < block_offset + number_of_blocks + 1; block++) {
        // read in block from global memory and store it in a memory efficient manner
        for (ulong row = 0; row < BLOCK_SIZE; row++) {
            for (ulong col = 0; col < BLOCK_SIZE / CHANNEL_WIDTH; col++) {
                if (block < number_of_blocks) {
                    load_chunk_of_a(A, a_block[block & 1], block, row, col);
                }
                if (block > 0) {
                    send_chunk_of_a{{ i }}(a_block[(block - 1) & 1], row, col);
                }
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
 * @param block_offset The first block that will be processed in the provided buffer
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose_write{{ i }}(__global DEVICE_DATA_TYPE *restrict B,
            __global DEVICE_DATA_TYPE *restrict A_out,
            const ulong block_offset,
            const ulong number_of_blocks) {

    #pragma loop_coalesce
    for (ulong current_block = block_offset; current_block < block_offset + number_of_blocks; current_block++) {
        // complete matrix transposition and write the result back to global memory
        for (ulong row = 0; row < BLOCK_SIZE; row++) {
            for (ulong col = 0; col < BLOCK_SIZE / CHANNEL_WIDTH; col++) {

                DEVICE_DATA_TYPE channel_data[CHANNEL_WIDTH];

                ch_data data1 = read_channel_intel(chan_a_in1{{ i }}); 
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH/2)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH/2; unroll_count++) {
                    channel_data[unroll_count] = data1.data[unroll_count];
                }

                ch_data data2 = read_channel_intel(chan_a_in2{{ i }}); 
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH/2)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH/2; unroll_count++) {
                    channel_data[CHANNEL_WIDTH/2 + unroll_count] = data2.data[unroll_count];
                }

                unsigned rot_out = row & (CHANNEL_WIDTH - 1);
                // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    A_out[current_block * BLOCK_SIZE * BLOCK_SIZE +
                    row * BLOCK_SIZE + col * CHANNEL_WIDTH + unroll_count] =
                    channel_data[unroll_count]
                    + B[current_block * BLOCK_SIZE * BLOCK_SIZE +
                    row * BLOCK_SIZE + col * CHANNEL_WIDTH + unroll_count];
                }
            }
        }
    }
}

{% endfor %}