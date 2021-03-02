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

#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define GEMM_BLOCK (1 << REGISTER_BLOCK_LOG)

#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct tmp_channel_chunk { DEVICE_DATA_TYPE data[GEMM_BLOCK];} ch_chunk_t;

// external channels from other devices
// depth is set to a single block row so calculation kernels do not need to stall
// until the network kernel has received everything
channel ch_chunk_t ch_top_in __attribute((io("kernel_input_ch0")));
channel ch_chunk_t ch_bottom_in __attribute((io("kernel_input_ch1")));
channel ch_chunk_t ch_left_in __attribute((io("kernel_input_ch2")));
channel ch_chunk_t ch_right_in __attribute((io("kernel_input_ch3")));

// external channels to other devices
// depth is set only to 1 because the receiver will buffer everything
channel ch_chunk_t ch_top_out __attribute((io("kernel_output_ch0")));
channel ch_chunk_t ch_bottom_out __attribute((io("kernel_output_ch1")));
channel ch_chunk_t ch_left_out __attribute((io("kernel_output_ch2")));
channel ch_chunk_t ch_right_out __attribute((io("kernel_output_ch3")));


// channels to and from the local kernels
channel ch_chunk_t ch_lu_col_out;
channel ch_chunk_t ch_lu_row_out;
channel ch_chunk_t ch_top_col_in;
channel ch_chunk_t ch_top_row_out;
channel ch_chunk_t ch_left_row_in;
channel ch_chunk_t ch_left_col_out;
channel ch_chunk_t ch_inner_row_in;
channel ch_chunk_t ch_inner_col_in;

/**
Takes care of the external channels.
Will forward data from calculation kernels to the external channels and will forward data if required.

operation_type: 0:inner, 1: left, 4:top, 8:lu and every combination of it
forward: 0: top, 1: right, 4: bottom, 8: left or every combination of it
 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void network_layer(
#ifdef EMULATE
	__global DEVICE_DATA_TYPE* restrict lu_scale_buffer,
#endif
				   const uint operation_type,
				   const uint forward_type) {

	DEVICE_DATA_TYPE current_scale;
#ifndef EMULATE
	// If not emulation, store LU row in local memory to get rid of stallable IO in this kernel
	DEVICE_DATA_TYPE lu_scale_buffer[BLOCK_SIZE];
#endif

	// For every row or column of the block, something needs to be sent
	#pragma loop_coalesce
	#pragma ivdep array(lu_scale_buffer)
	for (uint row = 0; row < BLOCK_SIZE; row++) {
		// Number of chunks that has to be processed
		#pragma ivdep array(lu_scale_buffer)
		for (uint chunk = 0; chunk < BLOCK_SIZE/GEMM_BLOCK; chunk++) {

			// Registers to store incoming and outgoing data chunks
			ch_chunk_t to_right;
			ch_chunk_t to_bottom;
			ch_chunk_t to_left;
			ch_chunk_t to_top;

			if ((operation_type & (LU_BLOCK_OUT)) && chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
				to_right = read_channel_intel(ch_lu_col_out);
				to_bottom = read_channel_intel(ch_lu_row_out);
				if (chunk == 0) {
					current_scale = to_right.data[row & (GEMM_BLOCK - 1)];
					lu_scale_buffer[row] = current_scale;
				}
			}
			// If LU block is not calculated on this FPGA
			// If left block, read from top and forward to bottom
			if (!(operation_type & (LU_BLOCK_OUT)) && ((forward_type & NETWORK_FWD_BOTTOM) || (operation_type & (LEFT_BLOCK))) && (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG))) {
				ch_chunk_t from_top = read_channel_intel(ch_top_in);
				// Forward chunk to the next top block
				to_bottom = from_top;
			}
			// If top block, read from left and forward to right
			if (!(operation_type & (LU_BLOCK_OUT)) && ((forward_type & NETWORK_FWD_RIGHT) || (operation_type & (TOP_BLOCK))) && (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG))) {
				ch_chunk_t from_left = read_channel_intel(ch_left_in);
				// Forward chunk to the next top block
				to_right = from_left;
				if (chunk == 0) {
					current_scale = from_left.data[row & (GEMM_BLOCK - 1)];
				}
			}
			if (!(operation_type & (LU_BLOCK_OUT)) && !(operation_type & (TOP_BLOCK)) && (operation_type & (TOP_BLOCK_OUT)) && chunk == 0) {
				current_scale = lu_scale_buffer[row];
			}
			//END LU block is not calculated on this FPGA

			if ((operation_type & (LEFT_BLOCK)) && chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
				write_channel_intel(ch_left_row_in, to_bottom);
			}
			if ((operation_type & (TOP_BLOCK)) && chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
				write_channel_intel(ch_top_col_in, to_right);
			}

			if (operation_type & (LEFT_BLOCK_OUT)) {
				to_left = read_channel_intel(ch_left_col_out);
			}
	
			// If inner block, receive from right and bottom and forward to left and top
			if (!(operation_type & (LEFT_BLOCK_OUT)) && (operation_type & (STORE_LEFT_INNER))) {
				ch_chunk_t from_right = read_channel_intel(ch_right_in);
				// Forward chunk to the next top block
				to_left = from_right;
			}


			if (operation_type & (STORE_LEFT_INNER)) {
				write_channel_intel(ch_inner_col_in, to_left);
			}

			if (operation_type & (TOP_BLOCK_OUT)) {
				ch_chunk_t from_top_kernel = read_channel_intel(ch_top_row_out);
				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					to_top.data[i] = from_top_kernel.data[i] * current_scale;
				}
			}
			// If inner block, receive from right and bottom and forward to left and top
			if (!(operation_type & (TOP_BLOCK_OUT)) && (operation_type & (STORE_TOP_INNER))) {
				ch_chunk_t from_bottom = read_channel_intel(ch_bottom_in);
				// Forward chunk to the next top block
				to_top = from_bottom;
			}

			if (operation_type & (STORE_TOP_INNER)) {
				write_channel_intel(ch_inner_row_in, to_top);
			}

			if ((forward_type & NETWORK_FWD_RIGHT) && chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
				write_channel_intel(ch_right_out, to_right);
			}
			if ((forward_type & NETWORK_FWD_BOTTOM) && chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
				write_channel_intel(ch_bottom_out, to_bottom);
			}
			if ((forward_type & NETWORK_FWD_LEFT)) {
				write_channel_intel(ch_left_out, to_left);
			}
			if ((forward_type & NETWORK_FWD_TOP)) {
				write_channel_intel(ch_top_out, to_top);
			}
		}
	}
}


/**
Executes a single step of the LU factorization.

This method takes a partially solved 8x8 matrix and calculates the next step of the LU factorization
The method needs 7 (GEMM_BLOCK-1) calls to perform a single LU factorization. This is done to reduce resource usage,
since all upcomng calls are anyway depending on the results of the previous call and there is no way
to pipeline multiple executions.

A is the input block that might be partially computed
step is the current step and must be a value between 0 to GEMM_BLOCK-2. After step GEMM_BLOCK-2, the block is factorized
 */
void
lu_block(const DEVICE_DATA_TYPE A[GEMM_BLOCK][GEMM_BLOCK], const int step, DEVICE_DATA_TYPE A_out[GEMM_BLOCK][GEMM_BLOCK]) {

	// Read current line from input
	DEVICE_DATA_TYPE line[GEMM_BLOCK];
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		line[i] = A[step][i];
	}

	// calculate the inverse of the diagonal element for the scaling
	DEVICE_DATA_TYPE inv_scale_a = -1.0 / line[step];

	// Scale the current row
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		if (i > step) {
			line[i] = line[i] * inv_scale_a;
		}
	}
	line[step] = inv_scale_a;

	// Update all rows fully unrolled
	// The multiply adds are fully independent
	//__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	// Unrolling disabled for this loop to save resources
	for (int j = 0; j < GEMM_BLOCK; j++) {
		DEVICE_DATA_TYPE curr_scale = A[j][step];
		// Update a single row. If it is already updated, just write back the value, if it is the current row
		// write back the value in "line", else update the value
		if (j != step) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i = 0; i < GEMM_BLOCK; i++) {
				A_out[j][i] = (i > step && j > step) ? A[j][i] + line[i] * curr_scale : A[j][i];
			}
		}
		else {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i = 0; i < GEMM_BLOCK; i++) {
				A_out[j][i] = line[i];
			}		
		}
	}
}

/**
This function can be used to update blocks using with three different operations.
It will execute the update for a single row in the block. The update is completed after GEMM_BLOCK calls of this
update function

operation_type: 0 for top = the top row of blocks will need a triangular MM
				1 for left = the left column of blocks will need a triangular MM, matrices have to be transposed
				2 for inner block == all inner blocks will be updated with a MM
 */
void
update_block(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK], 
			 const DEVICE_DATA_TYPE top[GEMM_BLOCK],
			 const DEVICE_DATA_TYPE left_or_lu[GEMM_BLOCK],
			 DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK],
			 const int current_row,
			 const int operation_type) {
	
	// Define different operation types of function
	const int op_top = 0;
	const int op_left = 1;
	const int op_inner = 2;

	// Transpose the input matrices if the target is a left block
	DEVICE_DATA_TYPE current_block[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register));
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = a[jj][ii] ;
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = a[ii][jj] ;
			}
		}
	}

	// Generate the first scalling array depending on the operation type
	DEVICE_DATA_TYPE scale_row[GEMM_BLOCK]  __attribute__((register));
	if (operation_type == op_inner) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] = top[jj];
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] = current_block[current_row][jj];
		}
	}
	if (operation_type == op_top) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] *= left_or_lu[current_row];
		}
	}

	DEVICE_DATA_TYPE tmp[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register));
	// scale all values with the pre calculated scaling array and the second input
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int ii =0; ii < GEMM_BLOCK; ii++) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			// left_or_lu_block are stored transposed to simplify the data access here
			tmp[ii][jj] = current_block[ii][jj] + scale_row[jj] * left_or_lu[ii];
		}
	}

	// overwrite results that were calculated altough they are not needed for the triangular operations left and top
	if (operation_type != op_inner) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			if (ii == current_row) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					tmp[ii][jj] = scale_row[jj];
				}
			}
			else if (ii < current_row) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					tmp[ii][jj] = current_block[ii][jj];
				}				
			}
		}
	}

	// write result back and transpose if necessary
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				out[ii][jj] = tmp[jj][ii];
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				out[ii][jj] = tmp[ii][jj];
			}
		}		
	}
}

__attribute__((uses_global_work_offset(0)))
__kernel
void
lu(__global DEVICE_DATA_TYPE* restrict a, 
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	
	// Store current row and column in separate buffers for 
	// easier access in the deep pipeline
	// need to be declared as local to prevent the compiler from 
	local DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
	local DEVICE_DATA_TYPE left_buffer[2][GEMM_BLOCK];

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		// Read in current LU block
		DEVICE_DATA_TYPE lu_a_buffer_in[GEMM_BLOCK][GEMM_BLOCK];
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				lu_a_buffer_in[ii][jj] = a_buffer[k][k][ii][jj];
			}
		}

		DEVICE_DATA_TYPE lu_a_buffer_out[GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE lu_a_buffer_out_row[GEMM_BLOCK];
		DEVICE_DATA_TYPE lu_a_buffer_out_col[GEMM_BLOCK];
		// Calculate next row and column of LU factorization and store in local memory buffer
		lu_block(lu_a_buffer_in, kk, lu_a_buffer_out);
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				a_buffer[k][k][ii][jj] = lu_a_buffer_out[ii][jj];
			}
		}
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			lu_a_buffer_out_row[jj] = lu_a_buffer_out[kk][jj];
		}
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			lu_a_buffer_out_col[jj] = lu_a_buffer_out[jj][kk];
		}

		// The update pipeline does not need to be executed for the last
		// row of blocks
		if (gk < BLOCK_SIZE - GEMM_BLOCK) {

			// Update all other blocks with the new calculated row and column
			#pragma ivdep safelen(BLOCK_SIZE/GEMM_BLOCK)
			for (int ttj = 0; ttj < (BLOCK_SIZE/GEMM_BLOCK - k) * BLOCK_SIZE/GEMM_BLOCK; ttj++) {

				int j = (ttj) & (BLOCK_SIZE/GEMM_BLOCK - 1);
				int ti = (ttj + (k * BLOCK_SIZE/GEMM_BLOCK)) / (BLOCK_SIZE/GEMM_BLOCK);
				// always execute the pipeline for whole rows of matrix blocks.
				// Only execute update for blocks that are required.
				// This helps to keep constant latencies between data dependencies of the pipeline stages
				if (ti >= k && j >= k) {
					/*
					Update order of block is:
					First the block below the LU block
					Then the row of blocks right of LU block
					This way the left block of the next column will always be calculated in advance 
					because it will be needed as input for the subsequent blocks.
					*/
					int i = (j == k) ? ti + 1 : ti;

					// The last left block will be out of bounds because of the update strategy described above
					// Skip it
					if (i < BLOCK_SIZE/GEMM_BLOCK) {

						// TODO Split this up into three different styles to reduce logic usage?
						
						// copy the correct block in the second input buffer
						// this depends on the operations that has to be executed
						DEVICE_DATA_TYPE second_input[GEMM_BLOCK];
						if (j == k) {
							// left matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out_row[jj];
							}
						}
						else if (i == k) {
							// top matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out_col[jj];
							}
						}
						else {
							// inner block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = left_buffer[ti & 1][jj];
							}
						}
						DEVICE_DATA_TYPE a_input[GEMM_BLOCK][GEMM_BLOCK];
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_input[ii][jj] = a_buffer[i][j][ii][jj];
							}
						}
						DEVICE_DATA_TYPE top_input[GEMM_BLOCK];
						if (ttj >= BLOCK_SIZE/GEMM_BLOCK) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								top_input[jj] = top_buffer[j][jj];
							}
						}
						DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK] __attribute__((register));
						update_block(a_input, 
										top_input, 
										second_input, 
										out,
										kk,
										(i == k) ? 0 : ((j == k) ? 1 : 2));
						if (ttj < BLOCK_SIZE/GEMM_BLOCK) {
							// only update in the first row
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								top_buffer[ttj][jj] = out[kk][jj];
							}
						}
						if (i > k && j == k) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int ii =0; ii < GEMM_BLOCK; ii++) {
								left_buffer[(ti + 1) & 1][ii] = out[ii][kk];
							}
						}
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_buffer[i][j][ii][jj] = out[ii][jj];
							}
						}
					}
				}
			}
		}
		// Send current updated column to right neighbor
		for (int i = 0; i < (BLOCK_SIZE/GEMM_BLOCK - k); i++) {
			ch_chunk_t col_data;
			ch_chunk_t row_data;
			#pragma unroll GEMM_BLOCK
			for (int j = 0; j < GEMM_BLOCK; j++) {
				row_data.data[j] = a_buffer[k][i + k][kk][j];
				col_data.data[j] = a_buffer[i + k][k][j][kk];
			}
			write_channel_intel(ch_lu_col_out, col_data);
			write_channel_intel(ch_lu_row_out, row_data);
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

/**
Update the blocks to the right of the current LU block

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void top_update(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict lu_global_buffer,
				const uint is_first_block,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		DEVICE_DATA_TYPE current_lu_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_scale;

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			ch_chunk_t col_in;

			DEVICE_DATA_TYPE scale_chunk[GEMM_BLOCK];

			// get current row chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = a_buffer[k][col][kk][i];
			}

			// Store chunk for later update and forward it over external channel
			ch_chunk_t row_out;
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				row_out.data[i] = scale_chunk[i];
			}
			write_channel_intel(ch_top_row_out, row_out);
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				if (is_first_block) {
					col_in = read_channel_intel(ch_top_col_in);
					// Store received LU block transposed in global memory buffer to sustain between function calls
					#pragma unroll
					for (int i=0; i < GEMM_BLOCK; i++) {
						lu_global_buffer[gk * BLOCK_SIZE + col * GEMM_BLOCK + i] = col_in.data[i];
					}
				}
				else {
					// Load LU data from global memory instead of receiving it from the channel
					#pragma unroll
					for (int i=0; i < GEMM_BLOCK; i++) {
						col_in.data[i] = lu_global_buffer[gk * BLOCK_SIZE + col * GEMM_BLOCK + i];
					}
				}
				if (col == 0) {
					current_scale = col_in.data[kk];
				}
				#pragma unroll
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_col[col][i] = (col > 0 || i > kk) ? col_in.data[i] : 0.f;
				}
			}

			// scale current row chunk with the rows scale factor received over the external channel
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = scale_chunk[i] * current_scale;
			}

			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				current_row[col][i] = scale_chunk[i];
			}

			// Update local memory buffer with chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				a_buffer[k][col][kk][i] = scale_chunk[i];
			}
		}

		// Update all remaining rows
		#pragma loop_coalesce
		for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK - k; row++) {
			// Update whole rows!
			for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK; curr_col++) {
				DEVICE_DATA_TYPE colbuf[GEMM_BLOCK];
				#pragma unroll
				for (int j=0; j < GEMM_BLOCK; j++) {
					colbuf[j] = current_lu_col[row][j];
				}	
				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					#pragma unroll
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row + k][curr_col][i][j] += colbuf[i] * current_row[curr_col][j];
					}
				}
			}
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

/**
Update the blocks below the current LU block

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void left_update(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict lu_global_buffer,
				const uint is_first_block,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		DEVICE_DATA_TYPE current_lu_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			DEVICE_DATA_TYPE chunk[GEMM_BLOCK];
			// get current row chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				chunk[i] = a_buffer[col][k][i][kk];
			}

			// Store chunk for later update and forward it over external channel
			ch_chunk_t col_out;
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				col_out.data[i] = chunk[i];
				current_col[col][i] = chunk[i];
			}
			write_channel_intel(ch_left_col_out, col_out);

			ch_chunk_t row_in;
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				if (is_first_block) {
					row_in = read_channel_intel(ch_left_row_in);
					// Store received LU chunk in global memory buffer to sustain between function calls
					#pragma unroll
					for (int i=0; i < GEMM_BLOCK; i++) {
						lu_global_buffer[gk * BLOCK_SIZE + col * GEMM_BLOCK + i] = row_in.data[i];
					}
				}
				else {
					// Load LU data from global memory instead of receiving it from the channel
					#pragma unroll
					for (int i=0; i < GEMM_BLOCK; i++) {
						row_in.data[i] = lu_global_buffer[gk * BLOCK_SIZE + col * GEMM_BLOCK + i];
					}
				}
				#pragma unroll
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_row[col][i] = (col > 0 || i > kk) ? row_in.data[i] : 0.f;
				}
			}
		}

		// Update all rows
		#pragma loop_coalesce
		#pragma ivdep
		for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK; row++) {
			// Update only remaining row chunks
			#pragma ivdep
			for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK - k; curr_col++) {
				DEVICE_DATA_TYPE colbuf[GEMM_BLOCK];
				#pragma unroll
				for (int j=0; j < GEMM_BLOCK; j++) {
					colbuf[j] = current_col[row][j];
				}	
				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					#pragma unroll
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row][curr_col + k][i][j] += current_lu_row[curr_col][j] * colbuf[i];
					}
				}
			}
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}


 __attribute__((uses_global_work_offset(0)))
__kernel
void inner_store(__global DEVICE_DATA_TYPE* restrict left_buffer,
				__global DEVICE_DATA_TYPE* restrict top_buffer,
				const uint operation_type) {

	#pragma loop_coalesce
	for (int row = 0; row < BLOCK_SIZE; row++) {
		for (int chunk = 0; chunk < BLOCK_SIZE / GEMM_BLOCK; chunk++) {

			if (operation_type & STORE_LEFT_INNER) {
				// Store left buffer
				ch_chunk_t to_left = read_channel_intel(ch_inner_col_in);

				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					left_buffer[row * BLOCK_SIZE + chunk * GEMM_BLOCK + i] = to_left.data[i];
				}
			}
			if (operation_type & STORE_TOP_INNER) {
				// Store top buffer
				ch_chunk_t to_top = read_channel_intel(ch_inner_row_in);

				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					top_buffer[row * BLOCK_SIZE + chunk * GEMM_BLOCK + i] = to_top.data[i];
				}
			}
		}
	}		
}

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]

/**
Update the inner blocks using the left and right column and rows

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void inner_update_mm/*PY_CODE_GEN i*/(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict left_global_buffer,
				__global DEVICE_DATA_TYPE* restrict top_global_buffer,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	local DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	local DEVICE_DATA_TYPE left_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];

	// Load blocks to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					top_buffer[i][j][ii][jj] = top_global_buffer[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					left_buffer[i][j][ii][jj] = left_global_buffer[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
			}
		}
	}

	// Update whole block
	#pragma ivdep array(a_buffer) safelen((BLOCK_SIZE/GEMM_BLOCK)*(BLOCK_SIZE/GEMM_BLOCK))
	for (int c = 0; c < (BLOCK_SIZE/GEMM_BLOCK) * (BLOCK_SIZE/GEMM_BLOCK) * (BLOCK_SIZE/GEMM_BLOCK); c++) {

		int mcol = c / ((BLOCK_SIZE/GEMM_BLOCK)*(BLOCK_SIZE/GEMM_BLOCK));
		int row = (c / (BLOCK_SIZE/GEMM_BLOCK)) & ((BLOCK_SIZE/GEMM_BLOCK) - 1);
		int curr_col = c & ((BLOCK_SIZE/GEMM_BLOCK) - 1);

		DEVICE_DATA_TYPE top_sub[GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE left_sub[GEMM_BLOCK][GEMM_BLOCK];

		#pragma unroll
		for (int i = 0; i < GEMM_BLOCK; i++) {
			#pragma unroll
			for (int j=0; j < GEMM_BLOCK; j++) {
				top_sub[i][j] = top_buffer[mcol][curr_col][i][j];
			}
		}

		#pragma unroll
		for (int i = 0; i < GEMM_BLOCK; i++) {
			#pragma unroll
			for (int j=0; j < GEMM_BLOCK; j++) {
				left_sub[i][j] = left_buffer[mcol][row][i][j];
			}
		}

		DEVICE_DATA_TYPE result_sub[GEMM_BLOCK][GEMM_BLOCK];
		#pragma unroll
		for (int k=0; k < GEMM_BLOCK; k++) {
			#pragma unroll
			for (int i = 0; i < GEMM_BLOCK; i++) {
				#pragma unroll
				for (int j = 0; j < GEMM_BLOCK; j++) {
					result_sub[i][j] = ((k > 0) ? result_sub[i][j] : a_buffer[row][curr_col][i][j]) + left_sub[k][i] * top_sub[k][j];
				}
			}
		}

		#pragma unroll
		for (int i = 0; i < GEMM_BLOCK; i++) {
			#pragma unroll
			for (int j=0; j < GEMM_BLOCK; j++) {
				a_buffer[row][curr_col][i][j] = result_sub[i][j];
			}
		}
	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

// PY_CODE_GEN block_end
