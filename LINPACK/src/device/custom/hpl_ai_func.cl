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

typedef struct tmp_block_type { DEVICE_DATA_TYPE data[GEMM_BLOCK][GEMM_BLOCK];} block_t;


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
lu(const DEVICE_DATA_TYPE A[GEMM_BLOCK][GEMM_BLOCK], const int step, DEVICE_DATA_TYPE A_out[GEMM_BLOCK][GEMM_BLOCK]) {

	// Read current line from input
	DEVICE_DATA_TYPE line[GEMM_BLOCK];
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		line[i] = A[step][i];
	}

	DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK];
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int j = 0; j < GEMM_BLOCK; j++) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int i = 0; i < GEMM_BLOCK; i++) {
			a_block[j][i] =  A[j][i];
		}
	}

	// calculate the inverse of the diagonal element for the scaling
	DEVICE_DATA_TYPE inv_scale_a = -1.0 / line[step];

	// Scale the current row
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		line[i] = (i > step) ? line[i] * inv_scale_a : ((i == step) ? inv_scale_a :  line[i]);
	}

	// Update all rows fully unrolled
	// The multiply adds are fully independent
	//__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	// Unrolliong disabled for this loop to save resources
	for (int j = 0; j < GEMM_BLOCK; j++) {
		DEVICE_DATA_TYPE curr_scale = a_block[j][step];
		// Update a single row. If it is already updated, just write back the value, if it is the current row
		// write back the value in "line", else update the value
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int i = 0; i < GEMM_BLOCK; i++) {
			A_out[j][i] = (j > step && i > step) ? a_block[j][i] + line[i] * curr_scale : ((step == j) ? line[i] : a_block[j][i]);
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
			 const DEVICE_DATA_TYPE top[GEMM_BLOCK][GEMM_BLOCK],
			 const DEVICE_DATA_TYPE left_or_lu[GEMM_BLOCK],
			 DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK],
			 const int current_row,
			 const int operation_type) {
	
	// Define different operation types of function
	const int op_top = 0;
	const int op_left = 1;
	const int op_inner = 2;

	// Read in the input blocks (lu block is already stored in variable)
	DEVICE_DATA_TYPE current_block_in[GEMM_BLOCK][GEMM_BLOCK];
	DEVICE_DATA_TYPE top_block[GEMM_BLOCK][GEMM_BLOCK];

	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int ii =0; ii < GEMM_BLOCK; ii++) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			current_block_in[ii][jj] = a[ii][jj];
			top_block[ii][jj] = top[ii][jj];
		}
	}

	// Transpose the input matrices if the target is a left block
	DEVICE_DATA_TYPE current_block[GEMM_BLOCK][GEMM_BLOCK];
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = current_block_in[jj][ii] ;
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = current_block_in[ii][jj] ;
			}
		}
	}

	// Generate the first scalling array depending on the operation type
	DEVICE_DATA_TYPE scale_row[GEMM_BLOCK];
	if (operation_type == op_inner) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] = top_block[current_row][jj];
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

	DEVICE_DATA_TYPE tmp[GEMM_BLOCK][GEMM_BLOCK];
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
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				tmp[ii][jj] = (ii > current_row) ? tmp[ii][jj] : ((ii == current_row) ? scale_row[jj] : current_block[ii][jj]);
			}
		}
	}

	// write result back and transpose if necessary
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int ii =0; ii < GEMM_BLOCK; ii++) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			out[ii][jj] = (operation_type == op_left) ? tmp[jj][ii] : tmp[ii][jj];
		}
	}
}

__attribute__((uses_global_work_offset(0)))
__kernel
void
gefa(__global DEVICE_DATA_TYPE* restrict a,
	unsigned n_blocks) {

	DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	#pragma disable_loop_pipelining
	for (int k = 0; k < BLOCK_SIZE/GEMM_BLOCK; k++) {
		// Read in current LU block
		DEVICE_DATA_TYPE lu_a_buffer_in[GEMM_BLOCK][GEMM_BLOCK];
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				lu_a_buffer_in[ii][jj] = a_buffer[k][k][ii][jj];
			}
		}

		// For each row in a small block do a LU factorization and directly apply row and column 
		// to other blocks
		#pragma disable_loop_pipelining
		for (int kk = 0; kk < GEMM_BLOCK; kk++) {

			DEVICE_DATA_TYPE lu_a_buffer_out[GEMM_BLOCK][GEMM_BLOCK];
			// Calculate next row and column of LU factorization and store in local memory buffer
			lu(lu_a_buffer_in, kk, lu_a_buffer_out);
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int ii =0; ii < GEMM_BLOCK; ii++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					lu_a_buffer_in[ii][jj] = lu_a_buffer_out[ii][jj];
					a_buffer[k][k][ii][jj] = lu_a_buffer_out[ii][jj];
				}
			}


			// Store current row and column in separate buffers for 
			// easier access in the deep pipeline
			DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
			DEVICE_DATA_TYPE left_buffer[2][GEMM_BLOCK][GEMM_BLOCK];

			// Update all other blocks with the new calculated row and column
			#pragma ivdep array(a_buffer)
			#pragma ivdep array(top_buffer) safelen(BLOCK_SIZE/GEMM_BLOCK)
			#pragma ivdep array(left_buffer) safelen(BLOCK_SIZE/GEMM_BLOCK)
			for (int ttj = 0; ttj < BLOCK_SIZE/GEMM_BLOCK * BLOCK_SIZE/GEMM_BLOCK; ttj++) {

				int j = ttj & (BLOCK_SIZE/GEMM_BLOCK - 1);
				int ti = ttj / (BLOCK_SIZE/GEMM_BLOCK);
				// always execute the pipeline for the whole matrix block.
				// Only execute update for blocks that are required.
				// This helps to keep constant latencies between the calculation of blocks
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
						
						// copy the correct block in the second input buffer
						// this depends on the operations that has to be executed
						DEVICE_DATA_TYPE second_input[GEMM_BLOCK];
						if (j == k) {
							// left matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out[kk][jj];
							}
						}
						else if (i == k) {
							// top matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out[jj][kk];
							}
						}
						else {
							// inner block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = left_buffer[ti & 1][jj][kk];
							}
						}

						DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK];
						update_block(a_buffer[i][j], 
										top_buffer[j], 
										second_input, 
										out,
										kk,
										(i == k) ? 0 : ((j == k) ? 1 : 2));
						if (i == k && j > k) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int ii =0; ii < GEMM_BLOCK; ii++) {
								__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
								for (int jj =0; jj < GEMM_BLOCK; jj++) {
									top_buffer[j][ii][jj] = out[ii][jj];
								}
							}
						}
						else if (i > k && j == k) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int ii =0; ii < GEMM_BLOCK; ii++) {
								__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
								for (int jj =0; jj < GEMM_BLOCK; jj++) {
									left_buffer[(ti + 1) & 1][ii][jj] = out[ii][jj];
								}
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
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}


__attribute__((uses_global_work_offset(0)))
__kernel
void
gesl(__global DEVICE_DATA_TYPE* restrict a, 
	__global DEVICE_DATA_TYPE* restrict b,
	unsigned n_blocks) {

    const int n = n_blocks * BLOCK_SIZE;

	// solve l*y = b
	// For each row in matrix
	for (int k = 0; k < n - 1; k++) {
		// For each row below add
		for (int i = k + 1; i < n; i++) {
			// add solved upper row to current row
			b[i] += b[k] * a[n * k + i];
		}
	}

	// now solve  u*x = y
	for (int k = n - 1; k >= 0; k--) {
		b[k] = b[k] * -a[n * k + k];
		for (int i = 0; i < k; i++) {
			b[i] -= b[k] * a[n * k + i];
		}
	}
}