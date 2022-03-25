/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/

#include "parameters.h"

/* PY_CODE_GEN 
try:
    kernel_param_attributes = generate_attributes(num_replications)
except:
    kernel_param_attributes = ["" for i in range(num_replications)]
*/

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]

/**
 * Read blocks of matrix A and transpose them in memory.
 * Write the block into an external channel.
 *
 * Will do the following:
 *
 * A -> trans(A) -> ext. ch
 *
 * @param A Buffer for matrix A
 * @param B Buffer for matrix B
 * @param A_out Buffer for result matrix
 * @param offset Offset in blocks that is used to read the current block of A. Since A is read column-wise
                on the block level, the whole matrix A might be written to global memory and the relevant columns
                need to be picked using this offset.
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 * @param width_in_blocks The with of matrix A in blocks
 * @param height_in_blocks The height of matix A in blocks
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose/*PY_CODE_GEN i*/(__global /*PY_CODE_GEN kernel_param_attributes[i]*/ DEVICE_DATA_TYPE *restrict A,
                                __global /*PY_CODE_GEN kernel_param_attributes[i]*/ DEVICE_DATA_TYPE *restrict B,
                                __global /*PY_CODE_GEN kernel_param_attributes[i]*/ DEVICE_DATA_TYPE *restrict A_out,
            const uint offset_a,
            const uint offset_b,
            const uint number_of_blocks,
            const uint width_in_blocks,
            const uint height_in_blocks) {

    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_block[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH] __attribute__((xcl_array_partition(complete,2)));
    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_plus_b_block[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH] __attribute__((xcl_array_partition(complete,2)));

    // transpose the matrix block-wise from global memory
    for (uint block = 0; block < number_of_blocks; block++) {
#ifdef INTEL_FPGA
        // Load A to local memory
        #pragma loop_coalesce
#endif
#ifdef XILINX_FPGA
#ifdef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(BLOCK_SIZE / CHANNEL_WIDTH)))
#endif
#endif
        for (uint row = 0; row < BLOCK_SIZE; row++) {
#ifdef XILINX_FPGA
#ifndef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(1)))
#endif
#endif
            for (uint col = 0; col < BLOCK_SIZE / CHANNEL_WIDTH; col++) {
                ulong block_row_a = (block + offset_a) / width_in_blocks;
                ulong block_col_a = (block + offset_a) % width_in_blocks;
                ulong ls_address_trans = block_col_a * BLOCK_SIZE * BLOCK_SIZE * height_in_blocks +
                            block_row_a * BLOCK_SIZE + 
                            row * BLOCK_SIZE * height_in_blocks;

                // read in block of A from global memory and store it in a memory efficient manner for transpose
                DEVICE_DATA_TYPE rotate_in[CHANNEL_WIDTH];

                // Blocks of a will be stored columnwise in global memory
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    rotate_in[unroll_count] = A[ls_address_trans + col * CHANNEL_WIDTH + unroll_count];
                }

                uint chunk = row * (BLOCK_SIZE / CHANNEL_WIDTH) + col;

                unsigned rot = (row) & (CHANNEL_WIDTH - 1);

                // rotate temporary buffer to store data into local buffer
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    // every block of (N / CHANNEL_WIDTH), rotates the index by 1
                    // store in double buffer
                    a_block[chunk][unroll_count] = rotate_in[(unroll_count + CHANNEL_WIDTH - rot)
                                                                                                & (CHANNEL_WIDTH - 1)];
                }
            }
        }

        // Read transposed A from local memory and add B 
#ifdef INTEL_FPGA
        // Load A to local memory
        #pragma loop_coalesce
#endif
#ifdef XILINX_FPGA
#ifdef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(BLOCK_SIZE / CHANNEL_WIDTH)))
#endif
#endif
        for (uint row = 0; row < BLOCK_SIZE; row++) {
#ifdef XILINX_FPGA
#ifndef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(1)))
#endif
#endif
            for (uint col = 0; col < BLOCK_SIZE / CHANNEL_WIDTH; col++) {
                ulong block_row = (block + offset_b) / width_in_blocks;
                ulong block_col = (block + offset_b) % width_in_blocks;
                ulong ls_address_row = block_row * BLOCK_SIZE * BLOCK_SIZE * width_in_blocks +
                        block_col * BLOCK_SIZE + 
                        row * BLOCK_SIZE * width_in_blocks;
                uint chunk = row * (BLOCK_SIZE / CHANNEL_WIDTH) + col;

                DEVICE_DATA_TYPE data_chunk[CHANNEL_WIDTH];
                DEVICE_DATA_TYPE rotate_out[CHANNEL_WIDTH];

                uint base = col * BLOCK_SIZE;
                uint offset = row / CHANNEL_WIDTH;

                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    unsigned rot = ((CHANNEL_WIDTH + unroll_count - row) * (BLOCK_SIZE / CHANNEL_WIDTH)) &
                                                                                                (BLOCK_SIZE - 1);
                    unsigned row_rotate = base + offset + rot;
                    rotate_out[unroll_count] = a_block[row_rotate][unroll_count];
                }

                unsigned rot_out = row & (CHANNEL_WIDTH - 1);

                // rotate temporary buffer to store data into local buffer
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    data_chunk[unroll_count] = rotate_out[(unroll_count + rot_out) & (CHANNEL_WIDTH - 1)];
                }

                // load tranposed A from global memory
                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    data_chunk[unroll_count] += B[ls_address_row + col * CHANNEL_WIDTH + unroll_count];
                }

                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    a_plus_b_block[chunk][unroll_count] = data_chunk[unroll_count];
                }
            }
        }
        // Write back result
#ifdef INTEL_FPGA
        // Load A to local memory
        #pragma loop_coalesce
#endif
#ifdef XILINX_FPGA
#ifdef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(BLOCK_SIZE / CHANNEL_WIDTH)))
#endif
#endif
        for (uint row = 0; row < BLOCK_SIZE; row++) {
#ifdef XILINX_FPGA
#ifndef XILINX_UNROLL_INNER_LOOPS
        __attribute__((xcl_pipeline_loop(1)))
#endif
#endif
            for (uint col = 0; col < BLOCK_SIZE / CHANNEL_WIDTH; col++) {
                ulong block_row = (block + offset_b) / width_in_blocks;
                ulong block_col = (block + offset_b) % width_in_blocks;
                ulong ls_address_row = block_row * BLOCK_SIZE * BLOCK_SIZE * width_in_blocks +
                        block_col * BLOCK_SIZE + 
                        row * BLOCK_SIZE * width_in_blocks;
                uint chunk = row * (BLOCK_SIZE / CHANNEL_WIDTH) + col;

                __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
                for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
                    A_out[ls_address_row + col * CHANNEL_WIDTH + unroll_count] = a_plus_b_block[chunk][unroll_count];
                }
            }
        }
    }
}

// PY_CODE_GEN block_end
