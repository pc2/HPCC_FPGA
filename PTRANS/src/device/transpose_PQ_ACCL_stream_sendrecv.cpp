/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/
#include "parameters.h"
#include "ap_int.h"
#include "ap_utils.h"
#include "ap_axi_sdata.h"
#include "accl_hls.h"


const int block_size = BLOCK_SIZE;
const int channel_width = CHANNEL_WIDTH;

/**
 * @brief Modulo operation that always produces positive values in range [0,op-1]. This is required for the PQ transpose algorithm and is different from the usual remainder calculation done with %!
 * 
 * @tparam T Data type used for the modulo operation.
 * @param number Number the modulo is calculated from
 * @param op Modulo operator
 * @return T number mod op
 */
template<typename T> 
T mod(T number, T op) {
    T result = number % op;
    // result >= op required for unsinged data types
    return (result < 0 || result >= op) ? op + result : result;
}


void transpose_block_transpose(const DEVICE_DATA_TYPE *A,
            DEVICE_DATA_TYPE a_block[][channel_width],
            const unsigned int offset_a,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks) {

#pragma HLS INTERFACE axis register both port=krnl2cclo

    // transpose the matrix block-wise from global memory
read_A:
    for (unsigned int row = 0; row < block_size; row++) {
read_A_line:
        for (unsigned int col = 0; col < block_size / channel_width; col++) {
#pragma HLS PIPELINE
            unsigned long block_row_a = (offset_a) / width_in_blocks;
            unsigned long block_col_a = (offset_a) % width_in_blocks;
            unsigned long ls_address_trans = block_col_a * block_size * block_size * height_in_blocks +
                        block_row_a * block_size + 
                        row * block_size * height_in_blocks;


            // read in block of A from global memory and store it in a memory efficient manner for transpose
            DEVICE_DATA_TYPE rotate_in[channel_width];
#pragma HLS ARRAY_PARTITION variable = rotate_in complete dim = 0

            // Blocks of a will be stored columnwise in global memory
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                rotate_in[unroll_count] = A[ls_address_trans + col * channel_width + unroll_count];
            }

            unsigned int chunk = row * (block_size / channel_width) + col;

            unsigned rot = (row) % (channel_width);

            // rotate temporary buffer to store data into local buffer
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                // every block of (N / channel_width), rotates the index by 1
                // store in double buffer
                a_block[chunk][unroll_count] = rotate_in[(unroll_count + channel_width - rot)
                                                                                            % (channel_width)];
            }
        }
    }
}

void transpose_block_forward(DEVICE_DATA_TYPE a_block[][channel_width],
            STREAM<stream_word> &krnl2cclo) {

read_A:
    for (unsigned int row = 0; row < block_size; row++) {
read_A_line:
        for (unsigned int col = 0; col < block_size / channel_width; col++) {
            DEVICE_DATA_TYPE data_chunk[channel_width];
#pragma HLS ARRAY_PARTITION variable = data_chunk complete dim = 0
            DEVICE_DATA_TYPE rotate_out[channel_width];
#pragma HLS ARRAY_PARTITION variable = rotate_out complete dim = 0

            unsigned int base = col * block_size;
            unsigned int offset = row / channel_width;

            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                unsigned rot = ((channel_width + unroll_count - row) * (block_size / channel_width)) %
                                                                                            (block_size);
                unsigned row_rotate = base + offset + rot;
                rotate_out[unroll_count] = a_block[row_rotate][unroll_count];
            }

            unsigned rot_out = row % (channel_width);

            // rotate temporary buffer to store data into local buffer
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                data_chunk[unroll_count] = rotate_out[(unroll_count + rot_out) % (channel_width)];
            }

            stream_word tmp;

            // load tranposed A from global memory
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                DEVICE_DATA_TYPE v = data_chunk[unroll_count];
                tmp.data((unroll_count + 1) * sizeof(DEVICE_DATA_TYPE)*8 - 1, unroll_count * sizeof(DEVICE_DATA_TYPE)*8) 
                        = *reinterpret_cast<ap_uint<sizeof(DEVICE_DATA_TYPE)*8>*>(&v);
            }
            tmp.dest = 9;
            tmp.last = 1;
            tmp.keep = -1;
            STREAM_WRITE(krnl2cclo,tmp);              
        }
    }
}

/**
 *
 * ext. channel -> trans(A) + B -> A_out
 *
 * @param B Buffer for matrix B
 * @param A_out Buffer for result matrix
 * @param offset Offset in blocks that is used to read the current block of A. Since A is read column-wise
                on the block level, the whole matrix A might be written to global memory and the relevant columns
                need to be picked using this offset.
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 * @param width_in_blocks The with of matrix A in blocks
 * @param height_in_blocks The height of matix A in blocks
 */
void transpose_block_receive(const DEVICE_DATA_TYPE *B,
                                 DEVICE_DATA_TYPE *A_out,
            const unsigned int offset_b,
            const unsigned int width_in_blocks,
            STREAM<stream_word> &cclo2krnl) {
#pragma HLS INTERFACE axis register both port=cclo2krnl

    // transpose the matrix block-wise from global memory
#pragma HLS loop_tripcount min=1 max=1024 avg=1
        // Read transposed A from local memory and add B 
read_B:
    for (unsigned int row = 0; row < block_size; row++) {
read_B_line:
        for (unsigned int col = 0; col < block_size / channel_width; col++) {
            unsigned long block_row = (offset_b) / width_in_blocks;
            unsigned long block_col = (offset_b) % width_in_blocks;
            unsigned long ls_address_row = block_row * block_size * block_size * width_in_blocks +
                    block_col * block_size + 
                    row * block_size * width_in_blocks;
            unsigned int chunk = row * (block_size / channel_width) + col;

            DEVICE_DATA_TYPE data_chunk[channel_width];
#pragma HLS ARRAY_PARTITION variable = data_chunk complete dim = 0

            stream_word tmp = STREAM_READ(cclo2krnl);

            // rotate temporary buffer to store data into local buffer
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                ap_uint<sizeof(DEVICE_DATA_TYPE)*8> v = tmp.data((unroll_count + 1) * sizeof(DEVICE_DATA_TYPE)*8 - 1, unroll_count * sizeof(DEVICE_DATA_TYPE)*8);
                data_chunk[unroll_count] = *reinterpret_cast<DEVICE_DATA_TYPE*>(&v);
            }

            // load tranposed A from global memory
            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                data_chunk[unroll_count] += B[ls_address_row + col * channel_width + unroll_count];
            }

            for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                A_out[ls_address_row + col * channel_width + unroll_count] = data_chunk[unroll_count];
            }
        }
    }
}

void transpose_read_sendrecv(const DEVICE_DATA_TYPE* A,
                const int* target_list,
                int pq_row, int pq_col, 
                int pq_width, int pq_height,
                int gcd, int least_common_multiple,
                int height_per_rank,
                int width_per_rank,
                STREAM<stream_word> &krnl2cclo) {

    // Begin algorithm from Figure 14 for general case
    int g = mod(pq_row - pq_col, gcd);
    int p = mod(pq_col + g, pq_width);
    int q = mod(pq_row - g, pq_height);

    for (int j = 0; j < least_common_multiple/pq_width; j++) {
        for (int i = 0; i < least_common_multiple/pq_height; i++) {
            // Determine sender and receiver rank of current rank for current communication step
            int send_rank = mod(p + i * gcd, pq_width) + mod(q - j * gcd, pq_height) * pq_width;

            for (int col = 0; col < least_common_multiple/pq_width; col++) {
                for (int row = 0; row < least_common_multiple/pq_height; row++) {
                    if (target_list[row * least_common_multiple/pq_width + col] == send_rank) {
                        for (int lcm_col = 0; lcm_col < (width_per_rank)/(least_common_multiple/pq_height); lcm_col++) {
                            for (int lcm_row = 0; lcm_row < (height_per_rank)/(least_common_multiple/pq_width); lcm_row++) {
                                unsigned int matrix_buffer_offset = (row + lcm_col * least_common_multiple/pq_height) + (col + lcm_row * least_common_multiple/pq_width) * width_per_rank;
                                DEVICE_DATA_TYPE a_block[block_size * block_size / channel_width][channel_width];
                                transpose_block_transpose(A, a_block, matrix_buffer_offset, width_per_rank, height_per_rank);
                                transpose_block_forward(a_block, krnl2cclo);
                            }
                        }
                    }
                }
            }
        }
    }
}

void transpose_write_sendrecv(const DEVICE_DATA_TYPE* B,
                    DEVICE_DATA_TYPE* C,
                const int* target_list,
                int pq_row, int pq_col, 
                int pq_width, int pq_height,
                int gcd, int least_common_multiple,
                int height_per_rank,
                int width_per_rank,
                STREAM<stream_word> &cclo2krnl) {

    // Begin algorithm from Figure 14 for general case
    int g = mod(pq_row - pq_col, gcd);
    int p = mod(pq_col + g, pq_width);
    int q = mod(pq_row - g, pq_height);
    for (int j = 0; j < least_common_multiple/pq_width; j++) {
        for (int i = 0; i < least_common_multiple/pq_height; i++) {

            int recv_rank = mod(p - i * gcd, pq_width) + mod(q + j * gcd, pq_height) * pq_width;

            for (int col = 0; col < least_common_multiple/pq_width; col++) {
                for (int row = 0; row < least_common_multiple/pq_height; row++) {
                    if (target_list[row * least_common_multiple/pq_width + col] == recv_rank) {
                        for (int lcm_row = 0; lcm_row < (height_per_rank)/(least_common_multiple/pq_width); lcm_row++) {
                            for (int lcm_col = 0; lcm_col < (width_per_rank)/(least_common_multiple/pq_height); lcm_col++) {
                                unsigned int matrix_buffer_offset = (row + lcm_col * least_common_multiple/pq_height) + (col + lcm_row * least_common_multiple/pq_width) * width_per_rank;
                                transpose_block_receive(B,C,matrix_buffer_offset,width_per_rank, cclo2krnl);
                            }
                        }
                    }
                }
            }
        }
    } 
}

