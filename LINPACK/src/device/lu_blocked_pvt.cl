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


/**
Calculate for the Level 2 block:

do_acc true:  c = c + a * b
do_acc false: c = a * b

where a,b,c are matrices of size GEMM_BLOCK.
Calculation itself is fully unrolled.
 */
void register_gemm(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK],
                    const DEVICE_DATA_TYPE b[GEMM_BLOCK][GEMM_BLOCK],
                    DEVICE_DATA_TYPE c_out[GEMM_BLOCK][GEMM_BLOCK],
                    const bool do_acc) {
#ifdef INTEL_FPGA
    /* 
     * For Intel devices do a cannon matrix multiplication. 
     * This leads to higher kernel frequencies and thus performance.
     * For Xilinx, this type of optimization does not work well, so a 
     * standard matrix multiplication is used instead
     */

    DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK + 1];
    DEVICE_DATA_TYPE b_block[GEMM_BLOCK + 1][GEMM_BLOCK];
    DEVICE_DATA_TYPE c_block[GEMM_BLOCK][GEMM_BLOCK];

    // Load block of matrix A and B and init C and reorder values
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            int k = (x + y) % GEMM_BLOCK;
            a_block[y][x] = a[y][k];
            b_block[y][x] = b[k][x];
            c_block[y][x] = 0;
        }
    }

    // Calculate result for 8x8 matrix
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int i=0;i<GEMM_BLOCK; i++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK;x++) {
            a_block[x][GEMM_BLOCK] = a_block[x][0];
            b_block[GEMM_BLOCK][x] = b_block[0][x];
        }
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for(int y=0; y < GEMM_BLOCK; y++) {
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int x=0; x<GEMM_BLOCK;x++) {
                c_block[y][x] += a_block[y][x] * b_block[y][x];
                a_block[y][x] = a_block[y][x + 1];
                b_block[y][x] = b_block[y + 1][x];
            }
        }
    }
    // Write back to BRAM and accumulate
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for(int y=0; y < GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            c_out[y][x] += c_block[y][x];
        }
    }

#else
    DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs
    DEVICE_DATA_TYPE b_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs
    DEVICE_DATA_TYPE c_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs

    // Load block of matrix A and B from BRAM to registers
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            a_block[y][x] = a[y][x];
            b_block[y][x] = b[y][x];
        }
    }

    // Calculate result for 8x8 matrix
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            DEVICE_DATA_TYPE sum = do_acc ? c_out[y][x]  : 0.f;
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int i=0; i<GEMM_BLOCK; i++) {
                sum += a_block[y][i] * b_block[i][x];
            }
            c_block[y][x] = sum;
        }
    }

    // Write back to BRAM
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for(int y=0; y < GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            c_out[y][x] = c_block[y][x];
        }
    }
#endif
}


/**
GEMM for the Level 1 Block (from BRAM to BRAM)

@param a_block input block from A matrix
@param b_block input block from B matrix
@param c_block result block to fill (each block will be passed in multiple times)
@param do_acc: accumulate into c_block (if false, reset to 0 at first write)
*/
void
local_gemm(const DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK],
           const DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                            [GEMM_BLOCK][GEMM_BLOCK],
           DEVICE_DATA_TYPE c_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]) {
/**
The BRAM matrix multiplication works differently for Intel and Xilinx.
For Intel the kernel calculates the complete result of an GEMM_BLOCKxGEMM_BLOCK 
matrix block in registers and writes it back to BRAM. Thus, k is the most inner loop.

For Xilinx, k is the outer loop and thus it will calculate parial results for all
GEMM_BLOCKxGEMM_BLOCK matrix block and write the partial result directly back
to BRAM.
 */
#ifdef INTEL_FPGA
    #pragma loop_coalesce 2
    // For each column in top block
    for (int i = 0; i < BLOCK_SIZE / GEMM_BLOCK; i++) {
        // For each element below it in current block
        for (int j = 0; j < BLOCK_SIZE / GEMM_BLOCK; j++) {
            // For Intel FPGA accumulate all partial results in registers
            // tmp_mul and only write back to BRAM once 
            DEVICE_DATA_TYPE tmp_mul[GEMM_BLOCK][GEMM_BLOCK];
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int ii = 0; ii < GEMM_BLOCK; ii++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                    tmp_mul[ii][jj] = 0;
                }
            }

            // For each diagonal element in left block
            for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
                // accumulate when working on following ks
                register_gemm(a_block[i][k], b_block[k][j],
                    tmp_mul, (k>0));
            }
            // Write back accumulated result to BRAM and accumulate if requested from outside
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for(int y=0; y < GEMM_BLOCK; y++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int x=0; x<GEMM_BLOCK; x++) {
                    c_block[i][j][y][x] = c_block[i][j][y][x] + tmp_mul[y][x];
                }
            }    
        }
    }
#else
    // For each diagonal element in left block
    for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
        // For each column in top block
        for (int i = 0; i < BLOCK_SIZE / GEMM_BLOCK; i++) {
            // For each element below it in current block
            for (int j = 0; j < BLOCK_SIZE / GEMM_BLOCK; j++) {
                // accumulate when requested from outside OR when working on following ks
                register_gemm(a_block[i][k], b_block[k][j],
                    c_block[i][j], (k>0));
            }
        }
    }
#endif
}



__attribute__((uses_global_work_offset(0)))
__kernel
void
gefa(__global DEVICE_DATA_TYPE* restrict a,
	__global int* restrict ipvt,
	unsigned n_blocks) {

	const int n = n_blocks * BLOCK_SIZE;

	// Calculate the LU decomposition in a block-wise fashion
	//
	//  ________________________
	//  |   DONE               |
	//  |    __________________|
	//  |   |Next block        |
	//  |   |                  |
	//  |   |                  |
	//  |   |                  |
	//  |   |                  |
	//  |   |                  |
	//  |   |                  |
	//  |   |                  |
	//  ------------------------
	//
	//
	for (int diagonal_block = 0; diagonal_block < n_blocks; diagonal_block++) {

		// For each next block calculate LU decompsition for next
		// BLOCK_SIZE rows and BLOCK_SIZE columns
		//
		//   ___________________
		//  |LU. . . . . . . . | < calculate LU result for rows
		//  |. . . . . . . . . | <
		//  |. .               |
		//  |. .               |
		//  |. .    This is    |
		//  |. .      left     |
		//  |. .    for MM     |
		//  |. .               |
		//  --------------------
		//   ^ ^
		//   calculate LU result for columns
		//
		for (int k = diagonal_block * BLOCK_SIZE; k < (diagonal_block + 1) * BLOCK_SIZE; k++) {

			DEVICE_DATA_TYPE scale_a= 0.f;
			DEVICE_DATA_TYPE max_val = 0.f; 
			int curr_ipvt = k;
			DEVICE_DATA_TYPE old_a = 0.f;
			// Find the maximum of the current row
			for (int row_block = k / UNROLL_COUNT; row_block < (diagonal_block + 1) * BLOCK_SIZE / UNROLL_COUNT; row_block++) {
				DEVICE_DATA_TYPE a_tmp[UNROLL_COUNT];
				DEVICE_DATA_TYPE a_tmp_abs[UNROLL_COUNT];
				int index_tmp[UNROLL_COUNT];

				// Read a chunk of a into the cache
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int chunk = 0; chunk < UNROLL_COUNT; chunk++) {
					int curr_id = (row_block) * UNROLL_COUNT + chunk;
					a_tmp[chunk] = (curr_id >= k) ? a[n*k + curr_id] : 0.f;
					a_tmp_abs[chunk] = fabs(a_tmp[chunk]);
					index_tmp[chunk] = curr_id;
					if (curr_id == k) {
						old_a = a_tmp[chunk];
					}
				}
#if UNROLL_COUNT > 1
				// Find local absolute maximum of chunk
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int stage = 2; stage <= UNROLL_COUNT; stage += stage) {
					__attribute__((opencl_unroll_hint(UNROLL_COUNT / 2)))
					for (int pair = 0; pair < UNROLL_COUNT / stage; pair++) {
						a_tmp_abs[pair] = (a_tmp_abs[pair] > a_tmp_abs[UNROLL_COUNT / stage + pair]) ? a_tmp_abs[pair] : a_tmp_abs[UNROLL_COUNT / stage + pair];
						a_tmp[pair] = (a_tmp_abs[pair] > a_tmp_abs[UNROLL_COUNT / stage + pair]) ? a_tmp[pair] : a_tmp[UNROLL_COUNT / stage + pair];
						index_tmp[pair] = (a_tmp_abs[pair] > a_tmp_abs[UNROLL_COUNT / stage + pair]) ? index_tmp[pair] : index_tmp[UNROLL_COUNT / stage + pair];
					}
				}
#endif
				// Update global absolute maximum if necessary
				if (a_tmp_abs[0] > max_val) {
					curr_ipvt =  index_tmp[0];
					scale_a =  a_tmp[0];
					max_val =  a_tmp_abs[0];
				}
			}
			ipvt[k] = curr_ipvt;
			
			DEVICE_DATA_TYPE inv_scale_a = -1.0 / scale_a;

			// Split the row into chunks to allow caching in local memory
			#pragma max_concurrency 1
			for (int row_block = diagonal_block; row_block < n_blocks; row_block++) {
				DEVICE_DATA_TYPE a_tmp[BLOCK_SIZE];

				// Read a chunk of a into the cache and exchange the pivot element
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int chunk = 0; chunk < BLOCK_SIZE; chunk++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk;
					a_tmp[chunk] = (curr_id != k && curr_id != curr_ipvt) 
									? ( (curr_id > k) ? a[k * n + curr_id] * inv_scale_a : a[k * n + curr_id])
									: ((curr_id == k) ? inv_scale_a : old_a * inv_scale_a);
				}

				// Update values of a and store them back to global memory
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int chunk = 0; chunk <  BLOCK_SIZE; chunk++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk;
					// scale current row
					a[k * n + curr_id] = a_tmp[chunk];
				}

				// Update remaining rows 
				// Updates all columns of the first block and just down to BLOCK_SIZE in the
				// other blocks. The remaining values are updated using matrix multiplication
				#pragma max_concurrency 1
				for (int j = k + 1; j < ((row_block == diagonal_block) ? n : (diagonal_block + 1) * BLOCK_SIZE); j++) {

					DEVICE_DATA_TYPE a_lower_tmp[BLOCK_SIZE];

					// Read in the pivot element ofr the current row
					// Position changes because it will be exchanged in the first iteration
					DEVICE_DATA_TYPE lower_scale_a = (row_block == diagonal_block) ? a[j*n + curr_ipvt] : a[j*n + k];
					
					// only needed in the first iteration to store old value of index k
					DEVICE_DATA_TYPE lower_old_a = (row_block == diagonal_block) ? a[j*n + k] : 0.f;

					// Read a chunk of a into the cache and exchange the pivot element
					__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
					for (int chunk = 0; chunk < BLOCK_SIZE; chunk++) {
						int curr_id = (row_block) * BLOCK_SIZE + chunk;
						a_lower_tmp[chunk] = (curr_id != k && curr_id != curr_ipvt) 
										? ( (curr_id > k) ? a[j * n + curr_id] +  a_tmp[chunk] * lower_scale_a : a[j * n + curr_id])
										: ((curr_id == k) ? lower_scale_a : ((row_block == diagonal_block) 
													? lower_old_a +  a_tmp[chunk] * lower_scale_a
													: a[j * n + curr_id] +  a_tmp[chunk] * lower_scale_a));
					}

					// store them back to global memory
					__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
					for (int chunk = 0; chunk <  BLOCK_SIZE; chunk++) {
						int curr_id = (row_block) * BLOCK_SIZE + chunk;
						a[j * n + curr_id] = a_lower_tmp[chunk];
					}

					// if exchange element not in the current chunk, 
					// it was not already updated and has to be updated in global memory
					if (row_block == diagonal_block && (row_block + 1) * BLOCK_SIZE <= curr_ipvt) {
						a[j*n + curr_ipvt] = lower_old_a;
					}
				}
			}
		}

		// Update the remaining matrix using matrix multiplication
		// and add the result to the current block
		//
		//   ___________________
		//  |LU |a   |b   |c   |
		//  |___|____|____|____|
		//  |1  |1*a |1*b | ...|
		//  |___|____|____|    |
		//  |2  |2*a |...      |
		//  |___|____|         |
		//  |3  |...           |
		//  |   |              |
		//  --------------------
		//
        // Update the remaining matrix using matrix multiplication
        // and add the result to the current block
        //
        //   ___________________
        //  |LU |a   |b   |c   |
        //  |___|____|____|____|
        //  |1  |1*a |1*b | ...|
        //  |___|____|____|    |
        //  |2  |2*a |         |
        //  |___|____|         |
        //  |3  |...           |
        //  |   |              |
        //  --------------------
        //
        #pragma loop_coalesce 2
        #pragma max_concurrency 1
        for (int inner_x_block = (diagonal_block + 1); inner_x_block < n_blocks; inner_x_block++) {
            for (int inner_y_block = (diagonal_block + 1); inner_y_block < ((inner_x_block == (diagonal_block + 1)) ? n_blocks : (diagonal_block + 2)); inner_y_block++) {
                DEVICE_DATA_TYPE current_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                // Load the current block
#pragma loop_coalesce 2
                for (int i = 0; i < BLOCK_SIZE ; i++) {
                    for (int j = 0; j < BLOCK_SIZE / UNROLL_COUNT; j++) {
                        DEVICE_DATA_TYPE current_reorder_buffer[UNROLL_COUNT];
__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                        for (int u = 0; u < UNROLL_COUNT; u++) {
                            current_reorder_buffer[u] = a[(inner_y_block * n + inner_x_block) * BLOCK_SIZE +
                                j * UNROLL_COUNT + u + i * n];
                        }
__attribute__((opencl_unroll_hint(UNROLL_COUNT/GEMM_BLOCK)))
                        for (int b = 0; b < UNROLL_COUNT/GEMM_BLOCK; b++) {
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                            for (int u = 0; u < GEMM_BLOCK; u++) {
                                current_block[i / GEMM_BLOCK][j * UNROLL_COUNT / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = current_reorder_buffer[b * GEMM_BLOCK + u];
                            }
                        }
                    }
                }    
                // calculate the result for the current block
                for (int k = 0; k <= diagonal_block; k++) {
                    DEVICE_DATA_TYPE left_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                            [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                    DEVICE_DATA_TYPE top_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                            [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));

                    // load the needed left and top block
                    #pragma loop_coalesce 2
                    for (int i = 0; i < BLOCK_SIZE ; i++) {
                        for (int j = 0; j < BLOCK_SIZE / UNROLL_COUNT; j++) {
                            DEVICE_DATA_TYPE left_reorder_buffer[UNROLL_COUNT];
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                            for (int u = 0; u < UNROLL_COUNT; u++) {
                                left_reorder_buffer[u] = a[(inner_y_block * n + k) * BLOCK_SIZE +
                                    j * UNROLL_COUNT + u + i * n];
                            }
    __attribute__((opencl_unroll_hint(UNROLL_COUNT/GEMM_BLOCK)))
                            for (int b = 0; b < UNROLL_COUNT/GEMM_BLOCK; b++) {
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                                for (int u = 0; u < GEMM_BLOCK; u++) {
                                    left_block[i / GEMM_BLOCK][j * UNROLL_COUNT / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = left_reorder_buffer[b * GEMM_BLOCK + u];
                                }
                            }
                        }
                    }
                    #pragma loop_coalesce 2
                    for (int i = 0; i < BLOCK_SIZE ; i++) {
                        for (int j = 0; j < BLOCK_SIZE / UNROLL_COUNT; j++) {
                            DEVICE_DATA_TYPE top_reorder_buffer[UNROLL_COUNT];
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                            for (int u = 0; u < UNROLL_COUNT; u++) {
                                top_reorder_buffer[u] = a[(k * n + inner_x_block) * BLOCK_SIZE +
                                    j * UNROLL_COUNT + u + i * n];
                            }
    __attribute__((opencl_unroll_hint(UNROLL_COUNT/GEMM_BLOCK)))
                            for (int b = 0; b < UNROLL_COUNT/GEMM_BLOCK; b++) {
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                                for (int u = 0; u < GEMM_BLOCK; u++) {
                                    top_block[i / GEMM_BLOCK][j * UNROLL_COUNT / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = top_reorder_buffer[b * GEMM_BLOCK + u];
                                }
                            }
                        }
                    }

                    local_gemm(left_block, top_block, current_block);
                }

                // write block back to main memory
                #pragma loop_coalesce 2
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE/UNROLL_COUNT; j++) {
    __attribute__((opencl_unroll_hint(UNROLL_COUNT)))
                        for (int u = 0; u < UNROLL_COUNT; u++) {
                            a[(inner_y_block * n + inner_x_block) * BLOCK_SIZE + j * UNROLL_COUNT + u
                                    + i * n] = current_block[i/GEMM_BLOCK][(j * UNROLL_COUNT + u)/GEMM_BLOCK][i & (GEMM_BLOCK - 1)][(j * UNROLL_COUNT + u) & (GEMM_BLOCK - 1)];
                        }
                    }
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
	__global unsigned* restrict ipvt, 
	unsigned n_blocks) {

	const unsigned n = n_blocks * BLOCK_SIZE;

	// solve l*y = b
	// For each row in matrix

	// Load first pivot element index
	int curr_ipvt = ipvt[0];
	// load values that have to be exchanged in b for the first row
	DEVICE_DATA_TYPE scale_b = b[curr_ipvt];
	DEVICE_DATA_TYPE old_b = b[0];
	int next_ipvt = 1;
	DEVICE_DATA_TYPE next_scale_b = 0.f;
	DEVICE_DATA_TYPE next_old_b = 0.f;

	// Go through every value in b
	#pragma max_concurrency 1
	for (int k = 0; k < n - 1; k++) {

		// Split the row into chunks to allow caching in local memory
		#pragma max_concurrency 1
		for (int row_block = k / BLOCK_SIZE; row_block < n / BLOCK_SIZE; row_block++) {
			DEVICE_DATA_TYPE b_tmp[BLOCK_SIZE];

			if (row_block == k / BLOCK_SIZE) {
				// store the next pivot element index
				// The value will be cached for the next iteration which allows to 
				// get rid of a single load operation for all upcoming iterations
				next_ipvt = ipvt[k + 1];
			}

			// Read a chunk of b into the cache and exchange the pivot element
			for (int chunk = 0; chunk < BLOCK_SIZE/UNROLL_COUNT; chunk++) {
				DEVICE_DATA_TYPE b_burst[UNROLL_COUNT];
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					b_burst[i] = b[curr_id];
				}
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					b_tmp[chunk * UNROLL_COUNT + i] = (curr_id != k && curr_id != curr_ipvt) ? b_burst[i] : ((curr_id == k) ? scale_b : old_b);
				}
			}

			// Update values of b and store them back to global memory
			for (int chunk = 0; chunk <  BLOCK_SIZE/ UNROLL_COUNT; chunk++) {
				DEVICE_DATA_TYPE a_burst[UNROLL_COUNT];
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					a_burst[i] = a[n * k + curr_id];
				}

				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					DEVICE_DATA_TYPE new_val = (curr_id > k) ? b_tmp[chunk * UNROLL_COUNT + i] +  scale_b * a_burst[i] : b_tmp[chunk * UNROLL_COUNT + i];
					b[curr_id] = new_val;
					if (curr_id == next_ipvt) {
						// cached for next iteration
						next_scale_b = new_val;
					}
					if (curr_id == k + 1) {
						// cached for next iteration
						next_old_b = new_val;
					}
				}
			}

			if (row_block + 1 == n / BLOCK_SIZE) {
				// use cached values in next iteration
				scale_b = next_scale_b;
				old_b = next_old_b;
				curr_ipvt = next_ipvt;
			}
		}
	}

	// now solve  u*x = y

	// load the current sclae value for b from global memory
	DEVICE_DATA_TYPE curr_b = b[n-1];
	DEVICE_DATA_TYPE ux_scale_b = 0.f;

	// for every value in b
	#pragma max_concurrency 1
	for (int k = n - 1; k >= 0; k--) {

		// Split the row into chunks to allow caching in local memory
		#pragma max_concurrency 1
		for (int row_block = 0; row_block <= (k >> LOCAL_MEM_BLOCK_LOG); row_block++) {
			DEVICE_DATA_TYPE b_tmp[BLOCK_SIZE];

			if (row_block == 0) {
				// scale current b value
				ux_scale_b = curr_b * a[n * k + k];
			}

			// Read a chunk of b into the cache
			for (int chunk = 0; chunk < BLOCK_SIZE/UNROLL_COUNT; chunk++) {
				DEVICE_DATA_TYPE b_burst[UNROLL_COUNT];
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					b_burst[i] = b[curr_id];
				}
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					b_tmp[chunk * UNROLL_COUNT + i] = b_burst[i];
				}
			}

			// scale all other values of b and write them back to global memory
			// TODO: With Vitis this pipeline has an II=16 because of non-aligned accesses
			//       to global memory(?) Why? Maybe because of ak,k loaded for scaling and the
			//		load in this pipeline? 
			for (int chunk = 0; chunk <  BLOCK_SIZE/UNROLL_COUNT; chunk++) {
				DEVICE_DATA_TYPE a_burst[UNROLL_COUNT];

				// read in a
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					a_burst[i] = a[n * k + curr_id];
				}

				// Update values
				__attribute__((opencl_unroll_hint(UNROLL_COUNT)))
				for (int i = 0; i < UNROLL_COUNT; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * UNROLL_COUNT + i;
					DEVICE_DATA_TYPE new_val = (curr_id < k) ? b_tmp[chunk * UNROLL_COUNT + i] +  ux_scale_b * a_burst[i] : (curr_id != k) ? b_tmp[chunk * UNROLL_COUNT + i] : -ux_scale_b;
					b[curr_id] = new_val;
					if (curr_id == k - 1) {
						// cache next scale value for next iteration
						curr_b = new_val;
					}
				}
			}
		}
	}
}
